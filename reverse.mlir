func @foo(%arg0: tensor<3x5xi32>) -> tensor<3x5xi32> {
  %init = linalg.init_tensor [3, 5] : tensor<3x5xi32>
  %0 = linalg_ext.reverse
         dimensions(dense<[0]> : tensor<1xi64>)
         {__internal_linalg_transform__ = "tiling_input"}
         ins(%arg0: tensor<3x5xi32>)
         outs(%init: tensor<3x5xi32>) : tensor<3x5xi32>
  return %0 : tensor<3x5xi32>
}
