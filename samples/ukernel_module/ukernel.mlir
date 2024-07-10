#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#translation = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64,
  {indexing_maps = [#map, #map1, #map2],
   loop_range = array<i64: 2048, 10240, 1280>,
   workgroup_tile_sizes = array<i64: 64, 128, 0>}>
module {
  func.func @matmul_transpose_b_ukernel(
      %arg0: memref<64x1280xf16, strided<[1280, 1], offset: ?>, #gpu.address_space<global>>,
      %arg1: memref<128x1280xf16, strided<[1280, 1], offset: ?>, #gpu.address_space<global>>,
      %arg2: memref<64x128xf32, strided<[10240, 1], offset: ?>, #gpu.address_space<global>>) attributes {translation_info = #translation} {
    linalg.matmul_transpose_b
      ins(%arg0, %arg1 : memref<64x1280xf16, strided<[1280, 1], offset: ?>, #gpu.address_space<global>>, memref<128x1280xf16, strided<[1280, 1], offset: ?>, #gpu.address_space<global>>)
      outs(%arg2 : memref<64x128xf32, strided<[10240, 1], offset: ?>, #gpu.address_space<global>>)
    return
  }
}
