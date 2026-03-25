// Dynamic-batch matmul + bias: out[?b,m,n] = sum_k(lhs[?b,m,k] * rhs[?b,k,n]) + bias[?b,m]
//
// Tests the dynamic-shape device-codegen path. The batch dimension is dynamic
// (represented as ?), with concrete values passed as push constants at runtime.
// The workgroup_count.so computes grid dimensions from these push constants.
//
// Step 0: Create temp directory for artifacts.
// RUN: rm -rf %t && mkdir -p %t
//
// Step 1: Generate reference output via IREE's CPU path.
//         iree-run-module handles dynamic shapes by providing concrete inputs.
//         Index args (batch dim = 4) must be passed alongside tensor args.
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:     --iree-hal-local-target-device-backends=llvm-cpu %s -o %t/ref.vmfb
// RUN: iree-run-module --device=local-task --module=%t/ref.vmfb \
// RUN:     --function=matmul_bias_dynamic \
// RUN:     --input=4x128x256xf32=1 --input=4 \
// RUN:     --input=4x256x512xf32=1 --input=4 \
// RUN:     --input=4x128xf32=1 --input=4 \
// RUN:     --input=4x128x512xf32 --input=4 \
// RUN:     --output=@%t/expected.npy
//
// Step 2: Compile to HSACO via device-codegen.
// RUN: iree-device-codegen --iree-gpu-test-target=gfx1100 \
// RUN:     --output-dir=%t %s
//
// Step 3: Launch on GPU and verify against CPU reference.
//         Inputs must match Step 1 exactly (=1 for data buffers).
//         Push constants carry the batch dim (4) for each tensor's ? dim.
//         One i32 per index arg (4 tensors × 1 dynamic dim each = 4 values).
// RUN: run-hip-kernel --artifacts=%t \
// RUN:     --input=4x128x256xf32=1 --input=4x256x512xf32=1 \
// RUN:     --input=4x128xf32=1 --input=4x128x512xf32 \
// RUN:     --push-constants=4,4,4,4 \
// RUN:     --expected_output=@%t/expected.npy \
// RUN:   | FileCheck %s
// CHECK: PASS

func.func @matmul_bias_dynamic(
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
  %matmul = linalg.generic {
      indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                       affine_map<(b, m, n, k) -> (b, k, n)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<?x128x256xf32>, tensor<?x256x512xf32>)
      outs(%fill : tensor<?x128x512xf32>) {
    ^bb0(%a : f32, %b : f32, %c : f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %mul, %c : f32
      linalg.yield %add : f32
    } -> tensor<?x128x512xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                       affine_map<(b, m, n) -> (b, m)>,
                       affine_map<(b, m, n) -> (b, m, n)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%matmul, %bias : tensor<?x128x512xf32>, tensor<?x128xf32>)
      outs(%empty : tensor<?x128x512xf32>) {
    ^bb0(%a : f32, %b : f32, %c : f32):
      %add = arith.addf %a, %b : f32
      linalg.yield %add : f32
    } -> tensor<?x128x512xf32>
  return %result : tensor<?x128x512xf32>
}
