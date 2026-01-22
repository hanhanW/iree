// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::Encoding {

//===----------------------------------------------------------------------===//
// encoding.set_encoding
//===----------------------------------------------------------------------===//

LogicalResult SetEncodingOp::verify() {
  // Source and the result have the same rank.
  if (getSourceType().getEncoding()) {
    return emitOpError(
        "source of set_encoding op cannot have a tensor encoding");
  }
  if (!isa_and_nonnull<SerializableAttr>(getResultType().getEncoding())) {
    return emitOpError(
        "result of set_encoding op expected to have a valid tensor encoding");
  }
  // The source and result must have the same rank.
  if (getResultType().getRank() != getSourceType().getRank()) {
    return emitOpError("cannot change the rank of the tensor");
  }
  if (failed(verifyCompatibleShape(getResultType(), getSourceType()))) {
    return emitOpError("expected to preserve the logical shape of the tensor");
  }
  return success();
}

LogicalResult SetEncodingOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] =
      tensor::getMixedSizes(builder, getLoc(), getSource());
  return success();
}

namespace {

/// Canonicalization pattern that replaces the encoding of a SetEncodingOp with
/// the serialized encoding from its DispatchTensorStoreOp users.
///
/// This pattern fires when:
/// 1. All users are DispatchTensorStoreOp
/// 2. All stores are full writes (not partial/tiled writes)
/// 3. All stores have the same serialized target encoding
///
/// This enables CSE to merge SetEncodingOps that have different abstract
/// encodings but resolve to the same physical layout after specialization.
struct PropagateSerializedEncodingFromStore
    : public OpRewritePattern<SetEncodingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SetEncodingOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      return rewriter.notifyMatchFailure(op, "no users");
    }

    SerializableAttr commonTargetEncoding;
    for (Operation *user : op->getUsers()) {
      auto storeOp = dyn_cast<TensorExt::DispatchTensorStoreOp>(user);
      if (!storeOp) {
        return rewriter.notifyMatchFailure(
            op, "has non-DispatchTensorStoreOp user");
      }
      if (!storeOp.isStoreToWholeTarget()) {
        return rewriter.notifyMatchFailure(
            op, "store is not a full write to the target");
      }

      auto encodingType =
          dyn_cast<EncodingTypeInterface>(storeOp.getTarget().getType());
      if (!encodingType) {
        return rewriter.notifyMatchFailure(
            op, "store target type does not implement EncodingTypeInterface");
      }
      auto targetEncoding =
          dyn_cast_if_present<SerializableAttr>(encodingType.getEncoding());
      if (!targetEncoding || !targetEncoding.isSerialized()) {
        return rewriter.notifyMatchFailure(
            op, "store target encoding is not serialized");
      }

      if (!commonTargetEncoding) {
        commonTargetEncoding = targetEncoding;
      } else if (commonTargetEncoding != targetEncoding) {
        return rewriter.notifyMatchFailure(
            op, "store users have different target encodings");
      }
    }

    if (op.getResultType().getEncoding() == commonTargetEncoding) {
      return rewriter.notifyMatchFailure(op,
                                         "encoding already matches target");
    }

    auto newResultType = RankedTensorType::get(
        op.getResultType().getShape(), op.getResultType().getElementType(),
        commonTargetEncoding);
    rewriter.modifyOpInPlace(op,
                             [&] { op.getResult().setType(newResultType); });
    return success();
  }
};

} // namespace

void SetEncodingOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<PropagateSerializedEncodingFromStore>(context);
}

//===----------------------------------------------------------------------===//
// encoding.unset_encoding
//===----------------------------------------------------------------------===//

LogicalResult UnsetEncodingOp::verify() {
  if (getResultType().getEncoding()) {
    return emitOpError(
        "result of unset_encoding op cannot have a tensor encoding");
  }
  if (!isa_and_nonnull<SerializableAttr>(getSourceType().getEncoding())) {
    return emitOpError(
        "source of unset_encoding op expected to have a valid tensor encoding");
  }
  // The source and result must have the same rank.
  if (getResultType().getRank() != getSourceType().getRank()) {
    return emitOpError("cannot change the rank of the tensor");
  }
  if (failed(verifyCompatibleShape(getResultType(), getSourceType()))) {
    return emitOpError("expected to preserve the logical shape of the tensor");
  }
  unsigned requiredDynCount = getResultType().getNumDynamicDims();
  if (getResultDims().size() != requiredDynCount) {
    return emitOpError() << "result type set has " << requiredDynCount
                         << " dynamic dimensions but only "
                         << getResultDims().size()
                         << " dimension values are attached";
  }
  return success();
}

LogicalResult UnsetEncodingOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] =
      getMixedValues(getResultType().getShape(), getResultDims(), builder);
  return success();
}

} // namespace mlir::iree_compiler::IREE::Encoding

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.cpp.inc" // IWYU pragma: keep
