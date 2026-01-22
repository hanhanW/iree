// RUN: iree-opt --allow-unregistered-dialect --split-input-file --canonicalize %s | FileCheck %s

// Test that SetEncodingOp's result type is updated to match the serialized
// encoding from its single DispatchTensorStoreOp user.

#serialized = #iree_encoding.identity
#unserialized = #iree_encoding.testing<>
// CHECK-LABEL: func.func @propagate_serialized_encoding_to_set_encoding
func.func @propagate_serialized_encoding_to_set_encoding(
    %arg0: tensor<?x?xf32>,
    %target: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>,
    %d0: index, %d1: index) {
  // CHECK: %[[SET:.+]] = iree_encoding.set_encoding %{{.+}} : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.identity>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #unserialized>
  // CHECK: iree_tensor_ext.dispatch.tensor.store %[[SET]]
  iree_tensor_ext.dispatch.tensor.store %0, %target, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32, #unserialized> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>{%d0, %d1}
  return
}

// -----

// Test with a serialized TestingAttr (has layouts).

// CHECK-DAG: #[[$SERIALIZED:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.identity]>
#serialized = #iree_encoding.testing<layouts = [#iree_encoding.identity]>
#unserialized = #iree_encoding.testing<>
// CHECK-LABEL: func.func @propagate_testing_with_layouts
func.func @propagate_testing_with_layouts(
    %arg0: tensor<16x32xf32>,
    %target: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x32xf32, #serialized>>) {
  // CHECK: %[[SET:.+]] = iree_encoding.set_encoding %{{.+}} : tensor<16x32xf32> -> tensor<16x32xf32, #[[$SERIALIZED]]>
  %0 = iree_encoding.set_encoding %arg0 : tensor<16x32xf32> -> tensor<16x32xf32, #unserialized>
  // CHECK: iree_tensor_ext.dispatch.tensor.store %[[SET]]
  iree_tensor_ext.dispatch.tensor.store %0, %target, offsets = [0, 0], sizes = [16, 32], strides = [1, 1]
      : tensor<16x32xf32, #unserialized> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x32xf32, #serialized>>
  return
}

// -----

// Test that the pattern DOES apply when SetEncodingOp has multiple store users
// with the same target encoding.

#serialized = #iree_encoding.identity
#unserialized = #iree_encoding.testing<>
// CHECK-LABEL: func.func @propagate_with_multiple_store_users
func.func @propagate_with_multiple_store_users(
    %arg0: tensor<?x?xf32>,
    %target0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>,
    %target1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>,
    %d0: index, %d1: index) {
  // CHECK: %[[SET:.+]] = iree_encoding.set_encoding %{{.+}} : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.identity>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #unserialized>
  // CHECK: iree_tensor_ext.dispatch.tensor.store %[[SET]], %{{.+}}
  iree_tensor_ext.dispatch.tensor.store %0, %target0, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32, #unserialized> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>{%d0, %d1}
  // CHECK: iree_tensor_ext.dispatch.tensor.store %[[SET]], %{{.+}}
  iree_tensor_ext.dispatch.tensor.store %0, %target1, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32, #unserialized> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>{%d0, %d1}
  return
}

// -----

// Test that the pattern does NOT apply when SetEncodingOp has a non-store user.

// CHECK-DAG: #[[$TESTING:.+]] = #iree_encoding.testing<>
#serialized = #iree_encoding.identity
#unserialized = #iree_encoding.testing<>
// CHECK-LABEL: func.func @no_propagation_non_store_user
func.func @no_propagation_non_store_user(
    %arg0: tensor<?x?xf32>,
    %target: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>,
    %d0: index, %d1: index) {
  // CHECK: iree_encoding.set_encoding %{{.+}} : tensor<?x?xf32> -> tensor<?x?xf32, #[[$TESTING]]>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #unserialized>
  iree_tensor_ext.dispatch.tensor.store %0, %target, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32, #unserialized> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>{%d0, %d1}
  // Non-store use prevents the pattern from applying.
  "test.sink"(%0) : (tensor<?x?xf32, #unserialized>) -> ()
  return
}

// -----

// Test that the pattern does NOT apply when store users have different target encodings.

// CHECK-DAG: #[[$TESTING:.+]] = #iree_encoding.testing<>
#serialized1 = #iree_encoding.identity
#serialized2 = #iree_encoding.testing<layouts = [#iree_encoding.identity]>
#unserialized = #iree_encoding.testing<>
// CHECK-LABEL: func.func @no_propagation_different_target_encodings
func.func @no_propagation_different_target_encodings(
    %arg0: tensor<?x?xf32>,
    %target0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized1>>,
    %target1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized2>>,
    %d0: index, %d1: index) {
  // CHECK: iree_encoding.set_encoding %{{.+}} : tensor<?x?xf32> -> tensor<?x?xf32, #[[$TESTING]]>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #unserialized>
  iree_tensor_ext.dispatch.tensor.store %0, %target0, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32, #unserialized> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized1>>{%d0, %d1}
  iree_tensor_ext.dispatch.tensor.store %0, %target1, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32, #unserialized> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized2>>{%d0, %d1}
  return
}

// -----

// Test that the pattern does NOT apply for partial writes (non-zero offsets).

// CHECK-DAG: #[[$TESTING:.+]] = #iree_encoding.testing<>
#serialized = #iree_encoding.identity
#unserialized = #iree_encoding.testing<>
// CHECK-LABEL: func.func @no_propagation_partial_write
func.func @no_propagation_partial_write(
    %arg0: tensor<8x16xf32>,
    %target: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x32xf32, #serialized>>) {
  // CHECK: iree_encoding.set_encoding %{{.+}} : tensor<8x16xf32> -> tensor<8x16xf32, #[[$TESTING]]>
  %0 = iree_encoding.set_encoding %arg0 : tensor<8x16xf32> -> tensor<8x16xf32, #unserialized>
  // Partial write with non-zero offset - pattern should NOT apply.
  iree_tensor_ext.dispatch.tensor.store %0, %target, offsets = [8, 16], sizes = [8, 16], strides = [1, 1]
      : tensor<8x16xf32, #unserialized> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x32xf32, #serialized>>
  return
}

// -----

// Test that the pattern does NOT apply when the single use is not DispatchTensorStoreOp.

// CHECK-DAG: #[[$TESTING:.+]] = #iree_encoding.testing<>
#unserialized = #iree_encoding.testing<>
// CHECK-LABEL: func.func @no_propagation_not_store_op
func.func @no_propagation_not_store_op(%arg0: tensor<?x?xf32>) {
  // CHECK: iree_encoding.set_encoding %{{.+}} : tensor<?x?xf32> -> tensor<?x?xf32, #[[$TESTING]]>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #unserialized>
  "test.sink"(%0) : (tensor<?x?xf32, #unserialized>) -> ()
  return
}

// -----

// Test that the pattern does NOT apply when the store target's encoding is not serialized.

// CHECK-DAG: #[[$TESTING:.+]] = #iree_encoding.testing<>
#unserialized1 = #iree_encoding.testing<>
#unserialized2 = #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32]>
// CHECK-LABEL: func.func @no_propagation_target_not_serialized
func.func @no_propagation_target_not_serialized(
    %arg0: tensor<?x?xf32>,
    %target: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #unserialized2>>,
    %d0: index, %d1: index) {
  // CHECK: iree_encoding.set_encoding %{{.+}} : tensor<?x?xf32> -> tensor<?x?xf32, #[[$TESTING]]>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #unserialized1>
  iree_tensor_ext.dispatch.tensor.store %0, %target, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32, #unserialized1> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #unserialized2>>{%d0, %d1}
  return
}

// -----

// Test that the pattern does NOT apply when the encodings are already the same.

#serialized = #iree_encoding.identity
// CHECK-LABEL: func.func @no_propagation_already_same
func.func @no_propagation_already_same(
    %arg0: tensor<?x?xf32>,
    %target: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>,
    %d0: index, %d1: index) {
  // CHECK: iree_encoding.set_encoding %{{.+}} : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.identity>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #serialized>
  iree_tensor_ext.dispatch.tensor.store %0, %target, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32, #serialized> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>{%d0, %d1}
  return
}

// -----

// Test with encoding_dims operands.

#serialized = #iree_encoding.identity
#unserialized = #iree_encoding.testing<>
// CHECK-LABEL: func.func @propagate_with_encoding_dims
func.func @propagate_with_encoding_dims(
    %arg0: tensor<?x?xf32>,
    %target: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>,
    %d0: index, %d1: index, %m: index, %n: index, %k: index) {
  // CHECK: %[[SET:.+]] = iree_encoding.set_encoding %{{.+}} encoding_dims{%{{.+}}, %{{.+}}, %{{.+}}} : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.identity>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #unserialized>
  // CHECK: iree_tensor_ext.dispatch.tensor.store %[[SET]]
  iree_tensor_ext.dispatch.tensor.store %0, %target, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32, #unserialized> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #serialized>>{%d0, %d1}
  return
}
