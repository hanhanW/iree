// RUN: iree-opt --split-input-file %s | FileCheck %s

func.func @test_empty_lowering_config() attributes {
    lowering_config = #iree_cpu.lowering_config<{}>} {
  return
}
// CHECK-LABEL: @test_empty_lowering_config()
// CHECK-SAME:    lowering_config = #iree_cpu.lowering_config<{}>

// -----

func.func @test_full_lowering_config() attributes {
    lowering_config = #iree_cpu.lowering_config<{
      distribution = [128, 128, 0],
      cache_parallel = [64, 64, 0],
      cache_reduction = [0, 0, 16],
      vector_common_parallel = [4, 4, 0],
      vector_reduction = [0, 0, 4],
      vector_inner_parallel = [0, 0, 0]
    }>} {
  return
}
// Order matters because it is sorted.
// CHECK-LABEL: @test_full_lowering_config()
// CHECK-SAME:    lowering_config = #iree_cpu.lowering_config<{
// CHECK-SAME:      cache_parallel = [64, 64, 0]
// CHECK-SAME:      cache_reduction = [0, 0, 16]
// CHECK-SAME:      distribution = [128, 128, 0]
// CHECK-SAME:      vector_common_parallel = [4, 4, 0]
// CHECK-SAME:      vector_inner_parallel = [0, 0, 0]
// CHECK-SAME:      vector_reduction = [0, 0, 4]
