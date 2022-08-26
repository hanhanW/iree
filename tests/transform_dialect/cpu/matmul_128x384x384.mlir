util.global private @"__iree_flow_128x384x384_lhs" {noinline} = dense<1.0> : tensor<128x384xf32>
util.global private @"__iree_flow_128x384x384_rhs" {noinline} = dense<1.0> : tensor<384x384xf32>
func.func @matmul_128x384x384() -> tensor<128x384xf32> {
    %lhs_ptr = util.global.address @"__iree_flow_128x384x384_lhs" : !util.ptr<tensor<128x384xf32>>
    %rhs_ptr = util.global.address @"__iree_flow_128x384x384_rhs" : !util.ptr<tensor<384x384xf32>>
    %lhs = util.global.load.indirect %lhs_ptr : !util.ptr<tensor<128x384xf32>> -> tensor<128x384xf32>
    %rhs = util.global.load.indirect %rhs_ptr : !util.ptr<tensor<384x384xf32>> -> tensor<384x384xf32>
    %init = linalg.init_tensor [128, 384] : tensor<128x384xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x384xf32>) -> tensor<128x384xf32>
    %0 = linalg.matmul
      ins(%lhs, %rhs: tensor<128x384xf32>, tensor<384x384xf32>)
      outs(%fill: tensor<128x384xf32>)
    -> tensor<128x384xf32>
    return %0: tensor<128x384xf32>
}

