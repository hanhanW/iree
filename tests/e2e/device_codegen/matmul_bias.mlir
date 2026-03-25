// Batched matmul + bias: out[b,m,n] = sum_k(lhs[b,m,k] * rhs[b,k,n]) + bias[b,m]
//
// This is the input format for the device-codegen-only pipeline.
// Output tensors are marked with {iree.abi.output = N}.
//
// Step 0: Create temp directory for artifacts.
// RUN: rm -rf %t && mkdir -p %t
//
// Step 1: Generate reference output via IREE's CPU path.
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:     --iree-hal-local-target-device-backends=llvm-cpu %s -o %t/ref.vmfb
// RUN: iree-run-module --device=local-task --module=%t/ref.vmfb \
// RUN:     --function=matmul_bias \
// RUN:     --input=4x128x256xf32=1 --input=4x256x512xf32=1 \
// RUN:     --input=4x128xf32=1 --input=4x128x512xf32 \
// RUN:     --output=@%t/expected.npy
//
// Step 2: Compile to HSACO via device-codegen.
// RUN: iree-device-codegen --iree-gpu-test-target=gfx1100 \
// RUN:     --output-dir=%t %s
//
// Step 3: Launch on GPU and verify against CPU reference.
//         Inputs must match Step 1 exactly (=1 for data buffers).
// RUN: run-hip-kernel --artifacts=%t \
// RUN:     --input=4x128x256xf32=1 --input=4x256x512xf32=1 \
// RUN:     --input=4x128xf32=1 --input=4x128x512xf32 \
// RUN:     --expected_output=@%t/expected.npy \
// RUN:   | FileCheck %s
// CHECK: PASS

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
  %matmul = linalg.generic {
      indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                       affine_map<(b, m, n, k) -> (b, k, n)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<4x128x256xf32>, tensor<4x256x512xf32>)
      outs(%fill : tensor<4x128x512xf32>) {
    ^bb0(%a : f32, %b : f32, %c : f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %mul, %c : f32
      linalg.yield %add : f32
    } -> tensor<4x128x512xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                       affine_map<(b, m, n) -> (b, m)>,
                       affine_map<(b, m, n) -> (b, m, n)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%matmul, %bias : tensor<4x128x512xf32>, tensor<4x128xf32>)
      outs(%empty : tensor<4x128x512xf32>) {
    ^bb0(%a : f32, %b : f32, %c : f32):
      %add = arith.addf %a, %b : f32
      linalg.yield %add : f32
    } -> tensor<4x128x512xf32>
  return %result : tensor<4x128x512xf32>
}
