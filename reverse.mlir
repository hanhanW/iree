func @foo(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  %init = linalg.init_tensor [5] : tensor<5xi32>
  %0 = linalg_ext.reverse
         dimension(0)
         {__internal_linalg_transform__ = "tiling_input"}
         ins(%arg0: tensor<5xi32>)
         outs(%init: tensor<5xi32>) : tensor<5xi32>
  return %0 : tensor<5xi32>
}
