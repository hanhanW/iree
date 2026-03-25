// Fully-dynamic matmul + bias: all dimensions are ?.
// out[?b,?m,?n] = sum_k(lhs[?b,?m,?k] * rhs[?b,?k,?n]) + bias[?b,?m]
//
// Tests the fully-dynamic device-codegen path where every dimension is
// runtime-determined via push constants. Concrete values: B=4, M=128, N=512, K=256.
//
// Step 0: Create temp directory for artifacts.
// RUN: rm -rf %t && mkdir -p %t
//
// Step 1: Generate reference output via IREE's CPU path.
//         All 15 arguments (4 tensors + 11 index dims) must be passed.
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:     --iree-hal-local-target-device-backends=llvm-cpu %s -o %t/ref.vmfb
// RUN: iree-run-module --device=local-task --module=%t/ref.vmfb \
// RUN:     --function=matmul_bias_fully_dynamic \
// RUN:     --input=4x128x256xf32=1 --input=4 --input=128 --input=256 \
// RUN:     --input=4x256x512xf32=1 --input=4 --input=256 --input=512 \
// RUN:     --input=4x128xf32=1 --input=4 --input=128 \
// RUN:     --input=4x128x512xf32 --input=4 --input=128 --input=512 \
// RUN:     --output=@%t/expected.npy
//
// Step 2: Compile to HSACO via device-codegen.
// RUN: iree-device-codegen --iree-gpu-test-target=gfx1100 \
// RUN:     --output-dir=%t %s
//
// Step 3: Launch on GPU and verify against CPU reference.
//         Inputs must match Step 1 exactly (=1 for data buffers).
//         Push constants: one i32 per index arg, in func signature order.
//         lhs: B=4,M=128,K=256  rhs: B=4,K=256,N=512
//         bias: B=4,M=128       out: B=4,M=128,N=512
// RUN: run-hip-kernel --artifacts=%t \
// RUN:     --input=4x128x256xf32=1 --input=4x256x512xf32=1 \
// RUN:     --input=4x128xf32=1 --input=4x128x512xf32 \
// RUN:     --push-constants=4,128,256,4,256,512,4,128,4,128,512 \
// RUN:     --expected_output=@%t/expected.npy \
// RUN:   | FileCheck %s
// CHECK: PASS

func.func @matmul_bias_fully_dynamic(
    %lhs : tensor<?x?x?xf32>,
    %lhs_d0 : index, %lhs_d1 : index, %lhs_d2 : index,
    %rhs : tensor<?x?x?xf32>,
    %rhs_d0 : index, %rhs_d1 : index, %rhs_d2 : index,
    %bias : tensor<?x?xf32>,
    %bias_d0 : index, %bias_d1 : index,
    %output : tensor<?x?x?xf32> {iree.abi.output = 0 : index},
    %out_d0 : index, %out_d1 : index, %out_d2 : index
) -> tensor<?x?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty(%out_d0, %out_d1, %out_d2) : tensor<?x?x?xf32>
  %fill = linalg.fill ins(%cst : f32)
      outs(%empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %matmul = linalg.generic {
      indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                       affine_map<(b, m, n, k) -> (b, k, n)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%fill : tensor<?x?x?xf32>) {
    ^bb0(%a : f32, %b : f32, %c : f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %mul, %c : f32
      linalg.yield %add : f32
    } -> tensor<?x?x?xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                       affine_map<(b, m, n) -> (b, m)>,
                       affine_map<(b, m, n) -> (b, m, n)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%matmul, %bias : tensor<?x?x?xf32>, tensor<?x?xf32>)
      outs(%empty : tensor<?x?x?xf32>) {
    ^bb0(%a : f32, %b : f32, %c : f32):
      %add = arith.addf %a, %b : f32
      linalg.yield %add : f32
    } -> tensor<?x?x?xf32>
  return %result : tensor<?x?x?xf32>
}
