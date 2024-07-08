func.func @main(%lhs: tensor<2048x1280xf16>, %rhs: tensor<10240x1280xf16>) -> tensor<2048x10240xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2048x10240xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
  %gemm = linalg.matmul_transpose_b
    ins(%lhs, %rhs: tensor<2048x1280xf16>, tensor<10240x1280xf16>)
    outs(%fill : tensor<2048x10240xf32>)
    -> tensor<2048x10240xf32>
  return %gemm : tensor<2048x10240xf32>
}
// RUN: iree-compile %s --iree-hal-target-backends=rocm \
// RUN:   --iree-rocm-target-chip=gfx90a \
// RUN:   --iree-codegen-mlir-ukernel-file-name=%p/ukernel.mlir \
// RUN:   --compile-to=executable-targets \
// RUN:   --mlir-disable-threading | \
// RUN: FileCheck %s
// TODO(hanchung): Add more checks. This is not done yet.
// CHECK: llvm.func @matmul_transpose_b_ukernel
// CHECK: llvm.func @main
