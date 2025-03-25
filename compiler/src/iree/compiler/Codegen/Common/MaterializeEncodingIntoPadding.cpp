// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATERIALIZEENCODINGINTOPADDINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using namespace IREE::Encoding;

namespace {

// Returns the pad encoding layout, or nullptr if this is not the only layout or
// if there's no encoding at all.
static PadEncodingLayoutAttr getPadLayout(Attribute layoutAttr,
                                          RankedTensorType type) {
  auto encoding =
      dyn_cast_or_null<IREE::Encoding::EncodingAttr>(type.getEncoding());
  if (!encoding) {
    return nullptr;
  }
  ArrayAttr layouts = encoding.getLayouts();
  if (!layouts) {
    return cast<PadEncodingLayoutAttr>(
        cast<IREE::Encoding::EncodingLayoutResolverAttrInterface>(layoutAttr)
            .getLayout(type));
  }
  if (layouts.size() != 1) {
    return nullptr;
  }

  return dyn_cast<PadEncodingLayoutAttr>(*layouts.begin());
}

// Returns a padded tensor type (without encoding) for tensor types with the pad
// encoding layout, or the same type for all other tensors.
static RankedTensorType getPaddedType(Attribute layoutAttr,
                                      RankedTensorType type) {
  PadEncodingLayoutAttr layout = getPadLayout(layoutAttr, type);
  if (!isNonZeroPadding(layout)) {
    return type.dropEncoding();
  }

  ArrayRef<int32_t> padding = layout.getPadding().asArrayRef();
  auto newShape = llvm::to_vector_of<int64_t>(type.getShape());
  for (auto [newDim, padValue] : llvm::zip_equal(newShape, padding)) {
    assert((padValue == 0 || !ShapedType::isDynamic(newDim)) &&
           "Padding dynamic dims not supported");
    newDim += padValue;
  }

  return RankedTensorType::get(newShape, type.getElementType());
}

// TODO(hanchung): Perhaps we can just check if the encoding is present and
// query if it is an identity layout.
static bool hasNonZeroPadding(Attribute layoutAttr, RankedTensorType type) {
  return isNonZeroPadding(getPadLayout(layoutAttr, type));
}

struct MaterializePadEncodingTypeConverter final
    : MaterializeEncodingTypeConverter {
  MaterializePadEncodingTypeConverter(
      IREE::Codegen::LayoutAttrInterface layoutAttr)
      : MaterializeEncodingTypeConverter(layoutAttr) {
    addConversion(
        [=](RankedTensorType type) -> std::optional<RankedTensorType> {
          if (!getPadLayout(layoutAttr, type)) {
            // Return `nullopt` so that other conversion functions have a chance
            // to handle this type.
            return std::nullopt;
          }
          return getPaddedType(layoutAttr, type);
        });
  }
};

/// Pattern to convert `flow.dispatch.tensor.load` operation when
/// materializing the encoding. We extract a smaller tensor for the padded
/// source. This way we do not create partial loads prematurely, which would be
/// difficult to undo later on.
struct MaterializeFlowDispatchTensorLoadOp final
    : OpMaterializeEncodingPattern<IREE::Flow::DispatchTensorLoadOp> {
  using OpMaterializeEncodingPattern::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchTensorLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the load covers the entire
    // `!flow.dispatch.tensor` type.
    if (!loadOp.isLoadOfWholeSource()) {
      return rewriter.notifyMatchFailure(loadOp, "unhandled partial loads");
    }

    auto &typeConverter =
        *getTypeConverter<MaterializePadEncodingTypeConverter>();
    IREE::Flow::DispatchTensorType sourceType = loadOp.getSourceType();
    auto boundTensorType = cast<RankedTensorType>(sourceType.getBoundType());
    if (!hasNonZeroPadding(typeConverter.getLayoutAttr(), boundTensorType)) {
      // Let the Nop pattern handle this.
      return rewriter.notifyMatchFailure(loadOp, "no padding applied");
    }

    auto paddedType =
        typeConverter.convertType<RankedTensorType>(boundTensorType);
    assert(paddedType != boundTensorType && "Expected conversion with padding");

    SmallVector<OpFoldResult> newMixedSizes =
        getMixedValues(paddedType.getShape(), loadOp.getSourceDims(), rewriter);

    SmallVector<OpFoldResult> newOffsets(newMixedSizes.size(),
                                         rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> newStrides(newMixedSizes.size(),
                                         rewriter.getIndexAttr(1));
    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newMixedSizes, newDynamicDims, newStaticDims);

    Location loc = loadOp.getLoc();
    Value newLoad = rewriter.create<IREE::Flow::DispatchTensorLoadOp>(
        loc, adaptor.getSource(), newDynamicDims, newOffsets, newMixedSizes,
        newStrides);
    rewriter.replaceOp(loadOp, newLoad);
    return success();
  }
};

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeFlowDispatchTensorStoreOp final
    : OpMaterializeEncodingPattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpMaterializeEncodingPattern::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchTensorStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the store covers the entire
    // `!flow.dispatch.tensor` type.
    if (!storeOp.isStoreToWholeTarget()) {
      return rewriter.notifyMatchFailure(storeOp, "unhandled partial stores");
    }

    auto &typeConverter =
        *getTypeConverter<MaterializePadEncodingTypeConverter>();
    IREE::Flow::DispatchTensorType targetType = storeOp.getTargetType();
    auto boundTensorType = cast<RankedTensorType>(targetType.getBoundType());
    if (!hasNonZeroPadding(typeConverter.getLayoutAttr(), boundTensorType)) {
      // Let the Nop pattern handle this.
      return rewriter.notifyMatchFailure(storeOp, "no padding applied");
    }

    auto paddedType =
        typeConverter.convertType<RankedTensorType>(boundTensorType);
    assert(paddedType != boundTensorType && "Expected conversion with padding");

    Location loc = storeOp.getLoc();
    SmallVector<OpFoldResult> offsets(paddedType.getRank(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(paddedType.getRank(),
                                      rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, adaptor.getValue());

    SmallVector<OpFoldResult> newMixedSizes = getMixedValues(
        paddedType.getShape(), storeOp.getTargetDims(), rewriter);
    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newMixedSizes, newDynamicDims, newStaticDims);

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, adaptor.getValue(), adaptor.getTarget(), newDynamicDims,
        offsets, newMixedSizes, strides);
    return success();
  }
};

/// Pattern to convert `set_encoding` op into `insert_slice` op. We create a
/// larger empty tensor for the destination and insert the value into it. This
/// way we do not create partial stores prematurely, which would be difficult to
/// undo later on.
struct SetEncodingOpLoweringConversion
    : public OpMaterializeEncodingPattern<IREE::Encoding::SetEncodingOp> {
  using OpMaterializeEncodingPattern<
      IREE::Encoding::SetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter =
        *getTypeConverter<MaterializePadEncodingTypeConverter>();
    RankedTensorType resultType = encodingOp.getResultType();

    Location loc = encodingOp.getLoc();
    int64_t rank = resultType.getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, adaptor.getSource());

    SmallVector<OpFoldResult> mixedResultSizes = sizes;
    PadEncodingLayoutAttr layout =
        getPadLayout(typeConverter.getLayoutAttr(), resultType);
    ArrayRef<int32_t> padding = layout.getPadding().asArrayRef();

    AffineExpr d0 = rewriter.getAffineDimExpr(0);
    AffineExpr d1 = rewriter.getAffineDimExpr(1);
    auto addMap = AffineMap::get(2, 0, {d0 + d1}, rewriter.getContext());
    for (auto [idx, value] : llvm::enumerate(padding)) {
      if (!value) {
        continue;
      }
      mixedResultSizes[idx] = affine::makeComposedFoldedAffineApply(
          rewriter, loc, addMap,
          ArrayRef<OpFoldResult>{mixedResultSizes[idx],
                                 rewriter.getIndexAttr(value)});
    }
    Value empty = rewriter.create<tensor::EmptyOp>(loc, mixedResultSizes,
                                                   resultType.getElementType());
    Value insertOp = rewriter.create<tensor::InsertSliceOp>(
        loc, adaptor.getSource(), empty, offsets, sizes, strides);
    rewriter.replaceOp(encodingOp, insertOp);

    return success();
  }
};

struct UnsetEncodingOpLoweringConversion
    : public OpMaterializeEncodingPattern<IREE::Encoding::UnsetEncodingOp> {
  using OpMaterializeEncodingPattern<
      IREE::Encoding::UnsetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Encoding::UnsetEncodingOp unsetEncodingOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = unsetEncodingOp.getResultType();
    int rank = resultType.getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = getMixedValues(
        resultType.getShape(), unsetEncodingOp.getResultDims(), rewriter);
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        unsetEncodingOp, resultType, adaptor.getSource(), offsets, sizes,
        strides);
    return success();
  }
};

/// Generic pattern to convert an operation.
template <typename OpTy>
struct MaterializeDPSOperation : public OpMaterializeEncodingPattern<OpTy> {
  using OpMaterializeEncodingPattern<OpTy>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        this->getTypeConverter());
    IREE::Codegen::LayoutAttrInterface layoutAttr = converter->getLayoutAttr();
    SmallVector<Value> operands;
    operands.append(adaptor.getInputs().begin(), adaptor.getInputs().end());
    operands.append(adaptor.getOutputs().begin(), adaptor.getOutputs().end());
    SmallVector<Type> convertedResTypes;
    for (auto init : op.getDpsInits()) {
      convertedResTypes.push_back(converter->convertType(init.getType()));
    }
    Operation *newOp =
        layoutAttr.lowerOp(rewriter, op, convertedResTypes, operands);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct MaterializeEncodingIntoPaddingPass final
    : impl::MaterializeEncodingIntoPaddingPassBase<
          MaterializeEncodingIntoPaddingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, arith::ArithDialect,
                linalg::LinalgDialect, tensor::TensorDialect,
                IREE::Codegen::IREECodegenDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface operation = getOperation();

    auto materializeEncodingValueFn =
        [](RankedTensorType, OpBuilder &,
           Location) -> FailureOr<MaterializeEncodingValueInfo> {
      return failure();
    };

    // Retrieve the config from executable target attribute, if any. Otherwise,
    // retrieve the config from CLI GPU target and construct a virtual
    // configuration.
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(operation);
    DictionaryAttr targetConfig;
    if (targetAttr) {
      targetConfig = targetAttr.getConfiguration();
    } else {
      IREE::GPU::TargetAttr targetAttr = getCLGPUTarget(context);
      SmallVector<NamedAttribute> items;
      items.emplace_back(
          IREE::Encoding::kEncodingResolverAttrName,
          IREE::GPU::getHIPTargetEncodingLayoutAttr(targetAttr, "pad"));
      targetConfig = DictionaryAttr::get(context, items);
    }

    // The layoutAttr should come in without any target info attached to it,
    // so we need to clone the layout attrs with the configuration so it can
    // access the target info during materialization.
    //
    // Otherwise, fall back to the nop layout.
    IREE::Codegen::LayoutAttrInterface layoutAttr;
    if (targetConfig &&
        targetConfig.contains(IREE::Encoding::kEncodingResolverAttrName)) {
      layoutAttr = targetConfig.getAs<IREE::Codegen::LayoutAttrInterface>(
          IREE::Encoding::kEncodingResolverAttrName);
      auto resolverAttr =
          cast<IREE::Encoding::EncodingLayoutResolverAttrInterface>(layoutAttr);
      layoutAttr = cast<IREE::Codegen::LayoutAttrInterface>(
          resolverAttr.cloneWithSimplifiedConfig(targetConfig));
    } else {
      layoutAttr = cast<IREE::Codegen::LayoutAttrInterface>(
          IREE::Codegen::EncodingNopLayoutAttr::get(context));
    }

    RewritePatternSet materializeEncodingPattern(context);
    MaterializePadEncodingTypeConverter typeConverter(layoutAttr);
    MaterializeEncodingConversionTarget target(*context);
    populateMaterializeEncodingPatterns(materializeEncodingPattern, target,
                                        typeConverter,
                                        materializeEncodingValueFn);

    // The majority of this conversion is based on the 'Nop' materialization,
    // with the exception of a few ops that have to account for padding.
    // We add custom patterns with much higher priority to run before the
    // equivalent 'Nop' patterns.
    materializeEncodingPattern.add<MaterializeFlowDispatchTensorLoadOp,
                                   MaterializeFlowDispatchTensorStoreOp,
                                   MaterializeDPSOperation<linalg::FillOp>,
                                   MaterializeDPSOperation<linalg::GenericOp>,
                                   SetEncodingOpLoweringConversion,
                                   UnsetEncodingOpLoweringConversion>(
        context, typeConverter, materializeEncodingValueFn,
        PatternBenefit{100});

    if (failed(applyPartialConversion(operation, target,
                                      std::move(materializeEncodingPattern)))) {
      operation.emitOpError("materialization failed");
      return signalPassFailure();
    }

    // Add patterns to resolve dims ops and cleanups.
    {
      RewritePatternSet patterns(context);
      memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
      context->getOrLoadDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      // TODO: Drop these when we deprecate partial loads/stores.
      IREE::Flow::populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
          patterns, context);
      if (failed(applyPatternsGreedily(operation, std::move(patterns)))) {
        operation.emitOpError("folding patterns failed");
        return signalPassFailure();
      }
    }
  }
};
} // namespace

void addEncodingToPaddingPasses(FunctionLikeNest &passManager) {
  passManager.addPass(createMaterializeEncodingIntoPaddingPass)
      .addPass(createBufferizeCopyOnlyDispatchesPass)
      .addPass(createCanonicalizerPass);
}

} // namespace mlir::iree_compiler
