// RUN: iree-opt --iree-codegen-convert-func-to-hal-executable %s | FileCheck %s

// Test: dynamic-shape batched matmul + bias is converted to hal.executable
// with push constants for dynamic dimensions.

// CHECK-LABEL: hal.executable public @matmul_bias
// CHECK:   hal.executable.variant public @rocm_hsaco_fb
// CHECK:     hal.executable.export public @matmul_bias ordinal(0)
// CHECK-SAME:   count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (index, index, index)
// CHECK:       iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3, %arg4)
// CHECK:       hal.return
// CHECK:     builtin.module
// CHECK:       func.func @matmul_bias()
//              Push constants:
// CHECK:         hal.interface.constant.load {{.*}} ordinal(0) : i32
// CHECK:         arith.index_castui
// CHECK:         iree_tensor_ext.dispatch.workload.ordinal {{.*}}, 0
// CHECK:         hal.interface.constant.load {{.*}} ordinal(1) : i32
// CHECK:         arith.index_castui
// CHECK:         iree_tensor_ext.dispatch.workload.ordinal {{.*}}, 1
// CHECK:         hal.interface.constant.load {{.*}} ordinal(2) : i32
// CHECK:         hal.interface.constant.load {{.*}} ordinal(3) : i32
//              Bindings with dynamic dims:
// CHECK:         hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128x256xf32>>
// CHECK:         iree_tensor_ext.dispatch.tensor.load
// CHECK:         hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x256x512xf32>>
// CHECK:         iree_tensor_ext.dispatch.tensor.load
// CHECK:         hal.interface.binding.subspan {{.*}} binding(2) {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x128xf32>>
// CHECK:         iree_tensor_ext.dispatch.tensor.load
// CHECK:         hal.interface.binding.subspan {{.*}} binding(3) {{.*}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x128x512xf32>>
//              Computation and store:
// CHECK:         linalg.generic
// CHECK:         linalg.generic
// CHECK:         iree_tensor_ext.dispatch.tensor.store

func.func @matmul_bias(
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
