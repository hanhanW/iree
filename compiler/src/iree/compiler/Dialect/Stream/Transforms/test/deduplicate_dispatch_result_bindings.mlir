// RUN: iree-opt --split-input-file --iree-stream-deduplicate-dispatch-result-bindings %s | FileCheck %s

// Test: Basic deduplication of two result bindings storing the same value.

#encoding = #iree_encoding.identity

// CHECK-LABEL: stream.executable private @test_dispatch_basic
stream.executable private @test_dispatch_basic {
  stream.executable.export public @entry workgroups(%arg0: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    // After deduplication: 3 args (index + input binding + 1 output binding)
    // CHECK: func.func @entry(%{{.*}}: index, %[[INPUT:.*]]: !stream.binding, %[[OUTPUT:.*]]: !stream.binding)
    // CHECK-NOT: !stream.binding)
    func.func @entry(%arg0: index, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) {
      %c0 = arith.constant 0 : index
      %dim = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
      %input = stream.binding.subspan %arg1[%c0] : !stream.binding
          -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%dim}
      %output0 = stream.binding.subspan %arg2[%c0] : !stream.binding
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32, #encoding>>{%dim}
      %output1 = stream.binding.subspan %arg3[%c0] : !stream.binding
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32, #encoding>>{%dim}

      %loaded = iree_tensor_ext.dispatch.tensor.load %input, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%dim} -> tensor<?x32xf32>

      %encoded = iree_encoding.set_encoding %loaded : tensor<?x32xf32> -> tensor<?x32xf32, #encoding>

      // Both stores use the same value - one should be removed.
      // CHECK: iree_tensor_ext.dispatch.tensor.store
      // CHECK-NOT: iree_tensor_ext.dispatch.tensor.store
      iree_tensor_ext.dispatch.tensor.store %encoded, %output0, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : tensor<?x32xf32, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32, #encoding>>{%dim}
      iree_tensor_ext.dispatch.tensor.store %encoded, %output1, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : tensor<?x32xf32, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32, #encoding>>{%dim}
      return
    }
  }
}

// CHECK-LABEL: util.func public @main_basic
util.func public @main_basic(%input: !stream.resource<*>, %input_size: index, %dim: index)
    -> (!stream.resource<*>, !stream.resource<*>) {
  %output_size = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x32xf32, #encoding>{%dim} : index

  // After deduplication: 2 results instead of 3.
  // CHECK: %[[RESULT:.*]]:2 = stream.tensor.dispatch
  %results:3 = stream.tensor.dispatch on(#hal.device.affinity<@__device_0>)
      @test_dispatch_basic::@entry[%dim](%dim, %input) :
      (index, tensor<?x32xf32>{%dim} in !stream.resource<*>{%input_size})
      -> (tensor<?x32xf32, #encoding>{%dim} in !stream.resource<*>{%output_size},
          tensor<?x32xf32, #encoding>{%dim} in !stream.resource<*>{%output_size},
          tensor<?x32xf32, #encoding>{%dim} in !stream.resource<*>{%output_size})

  // After deduplication, results#1 and results#2 are replaced by the kept result.
  // Original: return results#0, results#2 -> After: return result#0, result#1
  // CHECK: util.return %[[RESULT]]#0, %[[RESULT]]#1
  util.return %results#0, %results#2 : !stream.resource<*>, !stream.resource<*>
}

// -----

// Test: No deduplication when values are different.

#encoding2 = #iree_encoding.identity

// CHECK-LABEL: stream.executable private @test_dispatch_no_dedup
stream.executable private @test_dispatch_no_dedup {
  stream.executable.export public @entry workgroups(%arg0: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    // No deduplication: still 4 args.
    // CHECK: func.func @entry(%{{.*}}: index, %{{.*}}: !stream.binding, %{{.*}}: !stream.binding, %{{.*}}: !stream.binding)
    func.func @entry(%arg0: index, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) {
      %c0 = arith.constant 0 : index
      %dim = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
      %input = stream.binding.subspan %arg1[%c0] : !stream.binding
          -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%dim}
      %output0 = stream.binding.subspan %arg2[%c0] : !stream.binding
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32, #encoding2>>{%dim}
      %output1 = stream.binding.subspan %arg3[%c0] : !stream.binding
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%dim}

      %loaded = iree_tensor_ext.dispatch.tensor.load %input, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%dim} -> tensor<?x32xf32>

      %encoded = iree_encoding.set_encoding %loaded : tensor<?x32xf32> -> tensor<?x32xf32, #encoding2>

      // Different values stored - no deduplication should happen.
      // CHECK: iree_tensor_ext.dispatch.tensor.store
      // CHECK: iree_tensor_ext.dispatch.tensor.store
      iree_tensor_ext.dispatch.tensor.store %encoded, %output0, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : tensor<?x32xf32, #encoding2> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32, #encoding2>>{%dim}
      iree_tensor_ext.dispatch.tensor.store %loaded, %output1, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%dim}
      return
    }
  }
}

// CHECK-LABEL: util.func public @main_no_dedup
util.func public @main_no_dedup(%input: !stream.resource<*>, %input_size: index, %dim: index)
    -> (!stream.resource<*>, !stream.resource<*>) {
  %output0_size = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x32xf32, #encoding2>{%dim} : index
  %output1_size = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x32xf32>{%dim} : index

  // No deduplication: still 2 results.
  // CHECK: %[[RESULT:.*]]:2 = stream.tensor.dispatch
  %results:2 = stream.tensor.dispatch on(#hal.device.affinity<@__device_0>)
      @test_dispatch_no_dedup::@entry[%dim](%dim, %input) :
      (index, tensor<?x32xf32>{%dim} in !stream.resource<*>{%input_size})
      -> (tensor<?x32xf32, #encoding2>{%dim} in !stream.resource<*>{%output0_size},
          tensor<?x32xf32>{%dim} in !stream.resource<*>{%output1_size})

  // CHECK: util.return %[[RESULT]]#0, %[[RESULT]]#1
  util.return %results#0, %results#1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// Test: No deduplication for readwrite bindings (in-out buffers).
// Even if they store the same value, they are different buffers that can't be merged.

#encoding3 = #iree_encoding.identity

// CHECK-LABEL: stream.executable private @test_dispatch_readwrite_no_dedup
stream.executable private @test_dispatch_readwrite_no_dedup {
  stream.executable.export public @entry workgroups(%arg0: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    // No deduplication for readwrite: still 3 binding args.
    // CHECK: func.func @entry(%{{.*}}: index, %{{.*}}: !stream.binding, %{{.*}}: !stream.binding)
    func.func @entry(%arg0: index, %arg1: !stream.binding, %arg2: !stream.binding) {
      %c0 = arith.constant 0 : index
      %dim = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
      // Both bindings are readwrite - they are in-out buffers.
      %inout0 = stream.binding.subspan %arg1[%c0] : !stream.binding
          -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x32xf32>>{%dim}
      %inout1 = stream.binding.subspan %arg2[%c0] : !stream.binding
          -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x32xf32>>{%dim}

      // Load from both.
      %loaded0 = iree_tensor_ext.dispatch.tensor.load %inout0, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x32xf32>>{%dim} -> tensor<?x32xf32>
      %loaded1 = iree_tensor_ext.dispatch.tensor.load %inout1, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x32xf32>>{%dim} -> tensor<?x32xf32>

      // Compute something using both inputs.
      %result = arith.addf %loaded0, %loaded1 : tensor<?x32xf32>

      // Store the same result back to both - should NOT be deduplicated.
      // CHECK: iree_tensor_ext.dispatch.tensor.store
      // CHECK: iree_tensor_ext.dispatch.tensor.store
      iree_tensor_ext.dispatch.tensor.store %result, %inout0, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x32xf32>>{%dim}
      iree_tensor_ext.dispatch.tensor.store %result, %inout1, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x32xf32>>{%dim}
      return
    }
  }
}

// -----

// Test: No deduplication when a binding has multiple stores.
// We require exactly one store per binding for safety.

#encoding4 = #iree_encoding.identity

// CHECK-LABEL: stream.executable private @test_dispatch_multiple_stores_no_dedup
stream.executable private @test_dispatch_multiple_stores_no_dedup {
  stream.executable.export public @entry workgroups(%arg0: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    // No deduplication because binding0 has multiple stores.
    // CHECK: func.func @entry(%{{.*}}: index, %{{.*}}: !stream.binding, %{{.*}}: !stream.binding, %{{.*}}: !stream.binding)
    func.func @entry(%arg0: index, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) {
      %c0 = arith.constant 0 : index
      %dim = iree_tensor_ext.dispatch.workload.ordinal %arg0, 0 : index
      %input = stream.binding.subspan %arg1[%c0] : !stream.binding
          -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%dim}
      %output0 = stream.binding.subspan %arg2[%c0] : !stream.binding
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32, #encoding4>>{%dim}
      %output1 = stream.binding.subspan %arg3[%c0] : !stream.binding
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32, #encoding4>>{%dim}

      %loaded = iree_tensor_ext.dispatch.tensor.load %input, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%dim} -> tensor<?x32xf32>

      %intermediate = iree_encoding.set_encoding %loaded : tensor<?x32xf32> -> tensor<?x32xf32, #encoding4>
      // First store to output0.
      iree_tensor_ext.dispatch.tensor.store %intermediate, %output0, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : tensor<?x32xf32, #encoding4> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32, #encoding4>>{%dim}

      %final = iree_encoding.set_encoding %loaded : tensor<?x32xf32> -> tensor<?x32xf32, #encoding4>

      // Second store to output0 - this makes output0 have 2 stores, skipping deduplication.
      // CHECK: iree_tensor_ext.dispatch.tensor.store
      // CHECK: iree_tensor_ext.dispatch.tensor.store
      // CHECK: iree_tensor_ext.dispatch.tensor.store
      iree_tensor_ext.dispatch.tensor.store %final, %output0, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : tensor<?x32xf32, #encoding4> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32, #encoding4>>{%dim}
      iree_tensor_ext.dispatch.tensor.store %final, %output1, offsets = [0, 0], sizes = [%dim, 32], strides = [1, 1]
          : tensor<?x32xf32, #encoding4> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32, #encoding4>>{%dim}
      return
    }
  }
}
