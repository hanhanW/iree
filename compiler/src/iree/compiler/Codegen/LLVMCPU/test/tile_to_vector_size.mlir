// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-to-vector-size)" --split-input-file %s | FileCheck %s

#config = #iree_cpu.lowering_config<vector_common_parallel = [10, 20, 0], vector_reduction = [0, 0, 30]>
func.func @matmul_bias_add(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul {lowering_config = #config}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func.func @matmul_bias_add(
//      CHECK:   scf.forall
//      CHECK:     linalg.fill
//      CHECK:     linalg.matmul
//      CHECK:     linalg.generic
