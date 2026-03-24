// RUN: iree-opt --iree-gpu-test-target=gfx1100 \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:     iree-codegen-convert-func-to-hal-executable, \
// RUN:     hal.executable(hal.executable.variant( \
// RUN:       builtin.module(iree-codegen-llvmgpu-configuration-pipeline), \
// RUN:       iree-codegen-linalg-to-rocdl-pipeline)))" \
// RUN:   %s | FileCheck %s

// E2E test: func.func → hal.executable → ROCDL LLVM dialect IR.
// Verifies that the full pipeline from user-provided func.func to
// ROCDL kernel code produces valid LLVM dialect IR.

// CHECK-LABEL: hal.executable public @matmul_bias
// CHECK: hal.executable.variant public @rocm_hsaco_fb
// CHECK:   hal.executable.export public @matmul_bias
// CHECK-SAME: count
//   Workgroup count should be resolved to constants for static shapes:
// CHECK:     hal.return
//   The inner function should be lowered to LLVM dialect with rocdl.kernel:
// CHECK:   llvm.func @matmul_bias
// CHECK-SAME: !llvm.ptr<1>
// CHECK-SAME: rocdl.kernel

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
