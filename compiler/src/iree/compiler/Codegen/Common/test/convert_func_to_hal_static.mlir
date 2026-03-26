// RUN: iree-opt --iree-codegen-convert-func-to-hal-executable %s | FileCheck %s

// Test: static-shape batched matmul + bias is converted to hal.executable.
// The module must carry an hal.executable.target attr for the pass.

// CHECK-LABEL: hal.executable public @matmul_bias
// CHECK:   hal.executable.variant public @rocm_hsaco_fb
// CHECK:     hal.executable.export public @matmul_bias ordinal(0)
// CHECK-SAME:   count(%arg0: !hal.device) -> (index, index, index)
// CHECK:       iree_tensor_ext.dispatch.workgroup_count_from_slice()
// CHECK:       hal.return
// CHECK:     builtin.module
// CHECK:       func.func @matmul_bias()
//              Bindings (binding 0 = lhs, 1 = rhs, 2 = bias, 3 = output):
// CHECK:         hal.interface.binding.subspan {{.*}} binding(0) {{.*}} flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128x256xf32>>
// CHECK:         iree_tensor_ext.dispatch.tensor.load
// CHECK:         hal.interface.binding.subspan {{.*}} binding(1) {{.*}} flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x256x512xf32>>
// CHECK:         iree_tensor_ext.dispatch.tensor.load
// CHECK:         hal.interface.binding.subspan {{.*}} binding(2) {{.*}} flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128xf32>>
// CHECK:         iree_tensor_ext.dispatch.tensor.load
// CHECK:         hal.interface.binding.subspan {{.*}} binding(3) {{.*}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x128x512xf32>>
//              Computation (matmul + bias add):
// CHECK:         linalg.fill
// CHECK:         linalg.generic
// CHECK:         linalg.generic
// CHECK:         iree_tensor_ext.dispatch.tensor.store
// CHECK:         return

module attributes {"hal.executable.target" = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.target_info = #iree_gpu.target<arch = "gfx1100", features = "", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [<WMMAR3_F32_16x16x16_F16>, <WMMAR3_F16_16x16x16_F16>, <WMMAR3_F32_16x16x16_BF16>, <WMMAR3_BF16_16x16x16_BF16>, <WMMAR3_I32_16x16x16_I8>], subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 8192, workgroup_memory_bank_count = 64>>, ukernels = "none"}>} {
func.func @matmul_bias(
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
