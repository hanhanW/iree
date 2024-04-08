// RUN: iree-opt --split-input-file --verify-diagnostics %s

module {
  func.func @export_config_invalid_type() attributes {
    // expected-error @+1 {{expected workgroup size to have atmost 3 entries}}
    export_config = #iree_codegen.export_config<workgroup_size = [4, 1, 1, 1]>
  } {
    return
  }
}

// -----

module {
  func.func @encoding_round_dims_to() attributes {
    // expected-error @+1 {{expected value to be a positive number}}
    encoding.round_dims_to = #iree_codegen.encoding.round_dims_to<0>
  } {
    return
  }
  // CHECK: #iree_codegen.encoding.round_dims_to<16>
}
