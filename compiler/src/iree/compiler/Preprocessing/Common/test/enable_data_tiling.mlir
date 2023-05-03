// RUN: iree-opt --split-input-file -iree-preprocessing-enable-data-tiling %s | FileCheck %s

func.func @matmul_384x512x128(%lhs: tensor<384x128xf32>, %rhs: tensor<128x512xf32>) -> tensor<384x512xf32> {
  %init = tensor.empty() : tensor<384x512xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x512xf32>) -> tensor<384x512xf32>
  %0 = linalg.matmul
    ins(%lhs, %rhs: tensor<384x128xf32>, tensor<128x512xf32>)
    outs(%fill: tensor<384x512xf32>)
  -> tensor<384x512xf32>
  return %0: tensor<384x512xf32>
}
// CHECK-LABEL: func.func @matmul_384x512x128
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]
// CHECK:         %[[DEST:.+]] = tensor.empty() : tensor<384x512xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins({{.+}}) outs(%[[DEST]]
// CHECK:         %[[PACK_LHS:.+]] = tensor.pack %[[LHS]]
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [16, 1]
// CHECK-SAME:      into %{{.+}} : tensor<384x128xf32> -> tensor<24x128x16x1xf32>
// CHECK:         %[[PACK_RHS:.+]] = tensor.pack %[[RHS]]
// CHECK-SAME:      outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 1]
// CHECK-SAME:      into %{{.+}} : tensor<128x512xf32> -> tensor<32x128x16x1xf32>
// CHECK:         %[[PACK_FILL:.+]] = tensor.pack %[[FILL]]
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [16, 16]
// CHECK-SAME:      into %{{.+}} : tensor<384x512xf32> -> tensor<24x32x16x16xf32>
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:      ins(%[[PACK_LHS]], %[[PACK_RHS]]
// CHECK-SAME:      outs(%[[PACK_FILL]]
// CHECK:         %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<384x512xf32>
// CHECK:         %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]]
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [16, 16]
// CHECK-SAME:      into %[[UNPACK_DEST]] : tensor<24x32x16x16xf32> -> tensor<384x512xf32>

// -----

func.func @dynamic_matmul(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = linalg.matmul
    ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%fill: tensor<?x?xf32>)
  -> tensor<?x?xf32>
  return %0: tensor<?x?xf32>
}
// CHECK-LABEL: func.func @dynamic_matmul
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[DEST:.+]] = tensor.empty(%{{.+}}) : tensor<?x?xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins({{.+}}) outs(%[[DEST]]
// CHECK:         %[[PACK_LHS:.+]] = tensor.pack %[[LHS]]
// CHECK-SAME:      padding_value(%[[ZERO]] : f32) inner_dims_pos = [0, 1] inner_tiles = [16, 1]
// CHECK-SAME:      into %{{.+}} : tensor<?x?xf32> -> tensor<?x?x16x1xf32>
// CHECK:         %[[PACK_RHS:.+]] = tensor.pack %[[RHS]]
// CHECK-SAME:      padding_value(%[[ZERO]] : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 1]
// CHECK-SAME:      into %{{.+}} : tensor<?x?xf32> -> tensor<?x?x16x1xf32>
// CHECK:         %[[PACK_FILL:.+]] = tensor.pack %[[FILL]]
// CHECK-SAME:      padding_value(%[[ZERO]] : f32) inner_dims_pos = [0, 1] inner_tiles = [16, 16]
// CHECK-SAME:      into %{{.+}} : tensor<?x?xf32> -> tensor<?x?x16x16xf32>
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:      ins(%[[PACK_LHS]], %[[PACK_RHS]]
// CHECK-SAME:      outs(%[[PACK_FILL]]
// CHECK:         %[[UNPACK_DEST:.+]] = tensor.empty({{.+}}) : tensor<?x?xf32>
// CHECK:         %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]]
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [16, 16]
// CHECK-SAME:      into %[[UNPACK_DEST]] : tensor<?x?x16x16xf32> -> tensor<?x?xf32>
