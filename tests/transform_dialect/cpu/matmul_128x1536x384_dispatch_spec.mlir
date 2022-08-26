// RUN: iree-opt %s 

// Dispatch
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %foreach_thread, %tiled = 
      transform.structured.tile_to_foreach_thread_op %matmul tile_sizes [32, 32]
    transform.structured.fuse_into_containing_op %fill into %foreach_thread
    transform.iree.foreach_thread_to_flow %foreach_thread

    // transform.print {name = "AAA"}   
  }
}
