// RUN: iree-opt \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:     iree-codegen-convert-func-to-hal-executable, \
// RUN:     hal.executable(hal.executable.variant( \
// RUN:       builtin.module(iree-codegen-llvmgpu-configuration-pipeline), \
// RUN:       iree-codegen-linalg-to-rocdl-pipeline)), \
// RUN:     iree-codegen-extract-workgroup-count-as-func)" \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=STATIC

// RUN: iree-opt \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:     iree-codegen-convert-func-to-hal-executable, \
// RUN:     hal.executable(hal.executable.variant( \
// RUN:       builtin.module(iree-codegen-llvmgpu-configuration-pipeline), \
// RUN:       iree-codegen-linalg-to-rocdl-pipeline)), \
// RUN:     iree-codegen-extract-workgroup-count-as-func)" \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=DYNAMIC

// Test: extract workgroup count from static-shape matmul.
// After codegen, the count region has constant grid dims.
// The extracted func should return those constants with no arguments.

// STATIC-LABEL: hal.executable public @static_matmul
// STATIC: func.func @static_matmul_workgroup_count() -> (index, index, index)
// STATIC-SAME: attributes {subgroup_size = {{[0-9]+}} : index, workgroup_size = [{{[0-9]+}} : index, {{[0-9]+}} : index, {{[0-9]+}} : index]}
// STATIC:   %[[X:.+]] = arith.constant
// STATIC:   %[[Y:.+]] = arith.constant
// STATIC:   %[[Z:.+]] = arith.constant
// STATIC:   return %[[X]], %[[Y]], %[[Z]] : index, index, index

module attributes {"hal.executable.target" = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.target_info = #iree_gpu.target<arch = "gfx1100", features = "", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [<WMMAR3_F32_16x16x16_F16>, <WMMAR3_F16_16x16x16_F16>, <WMMAR3_F32_16x16x16_BF16>, <WMMAR3_BF16_16x16x16_BF16>, <WMMAR3_I32_16x16x16_I8>], subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 8192, workgroup_memory_bank_count = 64>>, ukernels = "none"}>} {
func.func @static_matmul(
    %lhs : tensor<4x128x256xf32>,
    %rhs : tensor<4x256x512xf32>,
    %bias : tensor<4x128xf32>,
    %output : tensor<4x128x512xf32> {iree.abi.output = 0 : index}
) -> tensor<4x128x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x128x512xf32>
  %fill = linalg.fill ins(%cst : f32)
      outs(%empty : tensor<4x128x512xf32>) -> tensor<4x128x512xf32>
  %gemm = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<4x128x256xf32>, tensor<4x256x512xf32>)
      outs(%fill : tensor<4x128x512xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %t0 = arith.mulf %b0, %b1 : f32
      %t1 = arith.addf %t0, %b2 : f32
      linalg.yield %t1 : f32
    } -> tensor<4x128x512xf32>
  %bias_add = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%gemm, %bias : tensor<4x128x512xf32>, tensor<4x128xf32>)
      outs(%empty : tensor<4x128x512xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %t0 = arith.addf %b0, %b1 : f32
      linalg.yield %t0 : f32
    } -> tensor<4x128x512xf32>
  return %bias_add : tensor<4x128x512xf32>
}
}

// -----

// Test: extract workgroup count from dynamic-shape matmul.
// After codegen, the count region has dynamic grid computation.
// The extracted func takes index args and computes grid dims.

// DYNAMIC-LABEL: hal.executable public @dynamic_matmul
// DYNAMIC: func.func @dynamic_matmul_workgroup_count(
// DYNAMIC-SAME: -> (index, index, index)
// DYNAMIC-SAME: workgroup_size = [{{[0-9]+}} : index, {{[0-9]+}} : index, {{[0-9]+}} : index]
// DYNAMIC:   return {{.*}} : index, index, index

module attributes {"hal.executable.target" = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.target_info = #iree_gpu.target<arch = "gfx1100", features = "", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [<WMMAR3_F32_16x16x16_F16>, <WMMAR3_F16_16x16x16_F16>, <WMMAR3_F32_16x16x16_BF16>, <WMMAR3_BF16_16x16x16_BF16>, <WMMAR3_I32_16x16x16_I8>], subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 8192, workgroup_memory_bank_count = 64>>, ukernels = "none"}>} {
func.func @dynamic_matmul(
    %lhs : tensor<?x128x256xf32>, %lhs_dim : index,
    %rhs : tensor<?x256x512xf32>, %rhs_dim : index,
    %bias : tensor<?x128xf32>, %bias_dim : index,
    %output : tensor<?x128x512xf32> {iree.abi.output = 0 : index},
    %output_dim : index
) -> tensor<?x128x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty(%output_dim) : tensor<?x128x512xf32>
  %fill = linalg.fill ins(%cst : f32)
      outs(%empty : tensor<?x128x512xf32>) -> tensor<?x128x512xf32>
  %gemm = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<?x128x256xf32>, tensor<?x256x512xf32>)
      outs(%fill : tensor<?x128x512xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %t0 = arith.mulf %b0, %b1 : f32
      %t1 = arith.addf %t0, %b2 : f32
      linalg.yield %t1 : f32
    } -> tensor<?x128x512xf32>
  %bias_add = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%gemm, %bias : tensor<?x128x512xf32>, tensor<?x128xf32>)
      outs(%empty : tensor<?x128x512xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %t0 = arith.addf %b0, %b1 : f32
      linalg.yield %t0 : f32
    } -> tensor<?x128x512xf32>
  return %bias_add : tensor<?x128x512xf32>
}
}
