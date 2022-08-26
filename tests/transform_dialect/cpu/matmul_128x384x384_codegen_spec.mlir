// RUN: iree-opt %s 

// Codegen
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):

    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1

    %tiled_matmul, %outer_loops:2 = transform.structured.fuse %matmul { tile_sizes=[32, 32, 0],
      tile_interchange = [0, 1, 2]}
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1
    %tiled_fill, %fill_loops:2 = transform.structured.tile %fill [4, 16] 
      {interchange = [0, 1]}
    %tiled_matmul_twice, %matmul_loops:3 = transform.structured.tile %tiled_matmul [16, 16, 1] 
      {interchange = [1, 0, 2]}


    // // transform.structured.pad %1 {
    // //     padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32], 
    // //     pack_paddings = [1, 1, 0],
    // //     hoist_paddings = [0, 0, 0],
    // //     transpose_paddings = [[1, 0], [0, 1]]
    // // }

    %func = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.vectorize %func {vectorize_padding = true}
    transform.iree.bufferize

    transform.lower_vectors {
      contraction_lowering = "outerproduct", 
      multireduction_lowering = "innerparallel", 
      split_transfers = "linalg-copy", 
      stages = [0, 1, 2, 3, 4, 5, 6, 7], 
      transpose_avx2_lowering = false, 
      transpose_lowering = "eltwise", 
      unroll_vector_transfers = true
    }

    // transform.print {name = "AAA"}


  }
}
