// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-llvmcpu-split-reduction"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUSPLITREDUCTIONPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

/// Make sure that
/// - the pass has not been applied before
/// - has tensor semantics
/// - number of reduction loops == 1
/// - has exactly 1 output
/// - index map has only projected permutations
/// - is a linalg generic op
/// - has exactly 1 input
/// - if enableReductionReordering is not set, then operand is an int
/// - innermost dimension of the input operand is reduction
/// TODO: support named ops, numInputs > 1, and modify lastDim check below
/// accordingly. If fpReductionReordering is not enabled by default, it must
/// be an integer or index type to proceed to allow associative reordering.
LogicalResult splitReductionPrecondition(Operation *op,
                                         bool fpReductionReordering) {
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);

  if (!linalgOp.hasPureTensorSemantics()) {
    LDBG() << "doesn't have tensor semantics";
    return failure();
  }
  if (linalgOp.getNumReductionLoops() != 1) {
    LDBG() << "number of reduction loops != 1";
    return failure();
  }
  if (linalgOp.getNumDpsInits() != 1) {
    LDBG() << "doesn't have exactly 1 output";
    return failure();
  }
  if (!linalgOp.hasOnlyProjectedPermutations()) {
    LDBG() << "index map doesn't have only projected permutations";
    return failure();
  }
  if (!isa<linalg::GenericOp>(op)) {
    LDBG() << "is not a generic op";
    return failure();
  }
  if (linalgOp.getNumDpsInputs() != 1) {
    LDBG() << "doesn't have exactly 1 input";
    return failure();
  }
  // The `linalg::splitReduction` method does not work for ops with indexing
  // semantics. See https://github.com/iree-org/iree/pull/14979
  if (linalgOp.hasIndexSemantics()) {
    LDBG()
        << "the split method used currently doesnt support indexing semantics";
    return failure();
  }

  auto elemType =
      getElementTypeOrSelf(linalgOp.getDpsInitOperand(0)->get().getType());
  if (!(fpReductionReordering || elemType.isIntOrIndex())) {
    LDBG() << "skipped because reduction reordering on FP is not enabled.";
    return failure();
  }

  SmallVector<unsigned> dims;
  linalgOp.getReductionDims(dims);
  AffineMap map =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  unsigned lastIdx = map.getNumResults() - 1;
  unsigned lastDim = map.getDimPosition(lastIdx);
  if (lastDim != dims[0]) {
    LDBG() << "innermost dimension of the input operand is not reduction";
    return failure();
  }

  return success();
}

/// Implements split reduction for bounded dynamic reduction dimensions.
/// Mirrors `linalg::splitReduction` but handles the case where the reduction
/// dimension is dynamic with a known upper bound that is a multiple of the
/// split size. Produces:
///   1. tensor.expand_shape: tensor<?xT> -> tensor<?x(splitSize)xT>
///   2. Partial reduction generic (outer reduction + inner parallel)
///   3. Final reduction generic (reduce the splitSize dim to scalar)
static FailureOr<linalg::SplitReductionResult>
splitReductionForBoundedDynamic(linalg::LinalgOp linalgOp, int64_t splitSize,
                                unsigned insertSplitIndex,
                                RewriterBase &rewriter) {
  SmallVector<unsigned> reductionDims;
  linalgOp.getReductionDims(reductionDims);
  unsigned reductionDim = reductionDims[0];
  // innerParallel=true: the new parallel dim is inserted after the reduction.
  unsigned insertSplitDimension = reductionDim + 1;

  // Find the operand dimension that maps to the reduction loop dimension.
  OpOperand *inputOperand = linalgOp.getDpsInputOperand(0);
  Value inputValue = inputOperand->get();
  auto inputType = cast<RankedTensorType>(inputValue.getType());
  AffineMap inputMap = linalgOp.getMatchingIndexingMap(inputOperand);

  int64_t reductionOperandDimIdx = -1;
  for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
    if (inputMap.getDimPosition(i) == reductionDim) {
      reductionOperandDimIdx = i;
      break;
    }
  }
  if (reductionOperandDimIdx < 0) {
    return failure();
  }

  int64_t reductionDimSize = inputType.getShape()[reductionOperandDimIdx];
  if (!ShapedType::isDynamic(reductionDimSize)) {
    return failure();
  }

  // Compute the upper bound of the dynamic reduction dimension.
  FailureOr<int64_t> ub = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, {inputValue, reductionOperandDimIdx},
      /*stopCondition=*/nullptr, /*closedUB=*/true);
  if (failed(ub)) {
    LDBG() << "could not compute upper bound for dynamic reduction dim";
    return failure();
  }

  if (ub.value() % splitSize != 0) {
    LDBG() << "upper bound " << ub.value()
           << " is not a multiple of split size " << splitSize;
    return failure();
  }

  // Match the reduction combiner and get its neutral element.
  SmallVector<Operation *, 4> combinerOps;
  if (!matchReduction(linalgOp.getRegionOutputArgs(), 0, combinerOps) ||
      combinerOps.size() != 1) {
    return failure();
  }

  Operation *reductionOp = combinerOps[0];
  std::optional<TypedAttr> identity = arith::getNeutralElement(reductionOp);
  if (!identity.has_value()) {
    return failure();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(linalgOp);
  Location loc = linalgOp.getLoc();

  // Build new inputs: reshape the reduction dim into [dynamic, splitSize].
  SmallVector<Value> newInputs;
  SmallVector<AffineMap> newMaps;
  for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
    AffineMap map = linalgOp.getMatchingIndexingMap(operand);
    SmallVector<int64_t> newShape;
    SmallVector<AffineExpr> exprs;
    SmallVector<ReassociationIndices> reassociation;
    unsigned index = 0;
    for (unsigned idx = 0; idx < map.getNumResults(); ++idx) {
      unsigned dim = map.getDimPosition(idx);
      if (reductionDim == dim) {
        // innerParallel: [outerReduction, innerParallel].
        newShape.push_back(ShapedType::kDynamic);
        newShape.push_back(splitSize);
        exprs.push_back(rewriter.getAffineDimExpr(
            dim < insertSplitDimension ? dim : dim + 1));
        exprs.push_back(rewriter.getAffineDimExpr(insertSplitDimension));
        reassociation.push_back({index++, index++});
        continue;
      }
      newShape.push_back(linalgOp.getShape(operand)[idx]);
      exprs.push_back(rewriter.getAffineDimExpr(
          dim < insertSplitDimension ? dim : dim + 1));
      reassociation.push_back({index++});
    }
    newMaps.push_back(
        AffineMap::get(map.getNumDims() + 1, 0, exprs, linalgOp.getContext()));

    if (newShape == SmallVector<int64_t>(linalgOp.getShape(operand))) {
      newInputs.push_back(operand->get());
      continue;
    }
    Type newType = RankedTensorType::get(
        newShape,
        cast<RankedTensorType>(operand->get().getType()).getElementType());
    Value newInput = tensor::ExpandShapeOp::create(
        rewriter, loc, newType, operand->get(), reassociation);
    newInputs.push_back(newInput);
  }

  // Build the intermediate output with the splitSize dimension inserted.
  SmallVector<int64_t> newOutputShape;
  AffineMap oldOutputMap =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
  ArrayRef<int64_t> oldShape = linalgOp.getShape(linalgOp.getDpsInitOperand(0));
  SmallVector<AffineExpr> outputExpr;
  for (unsigned idx = 0; idx <= oldShape.size(); ++idx) {
    if (insertSplitIndex == idx) {
      newOutputShape.push_back(splitSize);
      outputExpr.push_back(rewriter.getAffineDimExpr(insertSplitDimension));
    }
    if (idx < oldShape.size()) {
      newOutputShape.push_back(oldShape[idx]);
      unsigned dim = oldOutputMap.getDimPosition(idx);
      outputExpr.push_back(rewriter.getAffineDimExpr(
          dim < insertSplitDimension ? dim : dim + 1));
    }
  }

  Value emptyTensor =
      tensor::EmptyOp::create(rewriter, loc, newOutputShape,
                              linalgOp.getRegionOutputArgs()[0].getType());
  Value constantOp = arith::ConstantOp::create(rewriter, loc, *identity);
  auto fillOp = linalg::FillOp::create(rewriter, loc, constantOp, emptyTensor);
  Value identityTensor = fillOp.getResult(0);

  newMaps.push_back(AffineMap::get(oldOutputMap.getNumDims() + 1, 0, outputExpr,
                                   linalgOp.getContext()));

  // Iterator types: insert "parallel" at insertSplitDimension.
  SmallVector<utils::IteratorType> newIteratorTypes;
  for (auto [index, iteratorType] :
       llvm::enumerate(linalgOp.getIteratorTypesArray())) {
    if (insertSplitDimension == index) {
      newIteratorTypes.push_back(utils::IteratorType::parallel);
    }
    newIteratorTypes.push_back(iteratorType);
  }
  if (insertSplitDimension == linalgOp.getIteratorTypesArray().size()) {
    newIteratorTypes.push_back(utils::IteratorType::parallel);
  }

  // Create partial reduction generic.
  auto splitGenericOp = linalg::GenericOp::create(
      rewriter, loc, TypeRange({emptyTensor.getType()}), newInputs,
      ValueRange({identityTensor}), newMaps, newIteratorTypes);
  rewriter.inlineRegionBefore(linalgOp->getRegion(0),
                              splitGenericOp.getRegion(),
                              splitGenericOp.getRegion().begin());

  // Create final reduction generic (reduce the splitSize dimension).
  unsigned intermRank = newOutputShape.size();
  AffineMap finalInputMap = rewriter.getMultiDimIdentityMap(intermRank);
  SmallVector<utils::IteratorType> reductionIteratorTypes;
  SmallVector<AffineExpr> finalExprs;
  for (unsigned i = 0; i < intermRank; ++i) {
    if (insertSplitIndex == i) {
      reductionIteratorTypes.push_back(utils::IteratorType::reduction);
    } else {
      finalExprs.push_back(rewriter.getAffineDimExpr(i));
      reductionIteratorTypes.push_back(utils::IteratorType::parallel);
    }
  }
  AffineMap finalOutputMap =
      AffineMap::get(intermRank, 0, finalExprs, linalgOp.getContext());

  auto finalReduction = linalg::GenericOp::create(
      rewriter, loc, linalgOp->getResultTypes(),
      ValueRange({splitGenericOp.getResult(0)}), linalgOp.getDpsInits(),
      SmallVector<AffineMap>{finalInputMap, finalOutputMap},
      reductionIteratorTypes,
      [reductionOp](OpBuilder &b, Location loc, ValueRange inputs) {
        Operation *clonedReductionOp = b.clone(*reductionOp);
        clonedReductionOp->setOperand(0, inputs[0]);
        clonedReductionOp->setOperand(1, inputs[1]);
        linalg::YieldOp::create(b, loc, clonedReductionOp->getResult(0));
      });

  rewriter.replaceOp(linalgOp, finalReduction.getResults());

  return linalg::SplitReductionResult{
      emptyTensor.getDefiningOp(), fillOp,
      cast<linalg::LinalgOp>(splitGenericOp.getOperation()),
      cast<linalg::LinalgOp>(finalReduction.getOperation())};
}

/// Converts an inner-reduction into outer reduction + inner-parallel dimension,
/// followed by simple inner reduction.
LogicalResult splitReductionImpl(Operation *op, int64_t size,
                                 RewriterBase &rewriter) {
  IRRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);

  AffineMap map =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  unsigned lastIdx = map.getNumResults() - 1;
  linalg::ControlSplitReductionFn fn = [size, lastIdx](linalg::LinalgOp) {
    return linalg::SplitReductionOptions{size, lastIdx,
                                         /*innerParallel=*/true};
  };

  auto numLoops = linalgOp.getNumLoops();

  // 1) Tile to extract a single vector-length array.
  SmallVector<OpFoldResult> tileSizesSVFirst(numLoops,
                                             rewriter.getIndexAttr(1));
  tileSizesSVFirst[numLoops - 1] = rewriter.getIndexAttr(0);
  auto options = scf::SCFTilingOptions().setTileSizes(tileSizesSVFirst);
  FailureOr<scf::SCFTilingResult> tileResFirst = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(linalgOp.getOperation()), options);
  if (failed(tileResFirst)) {
    LDBG() << "failed on step 1 (SCFTiling)";
    return failure();
  }
  rewriter.replaceOp(linalgOp, tileResFirst->replacements);

  // 2) Apply splitReduction on the single vector-length array.
  // splitReduction already replaces the op.
  auto tiledOp = cast<linalg::LinalgOp>(tileResFirst->tiledOps.back());
  FailureOr<linalg::SplitReductionResult> splitRes =
      splitReduction(rewriter, tiledOp, fn);
  if (failed(splitRes)) {
    // Upstream splitReduction requires static shapes. Fall back to the
    // bounded-dynamic path which handles dynamic dims with known upper bounds.
    splitRes =
        splitReductionForBoundedDynamic(tiledOp, size, lastIdx, rewriter);
    if (failed(splitRes)) {
      LDBG() << "failed on step 2 (SplitReduction)";
      return success();
    }
  }

  // 3) Tile the first op generated by splitReduction with tile size of 1,
  // to essentially create a reduction loop. Note that
  // splitRes->splitLinalgOp.getNumLoops() = numLoops + 1.
  SmallVector<OpFoldResult> tileSizesSV(splitRes->splitLinalgOp.getNumLoops(),
                                        rewriter.getIndexAttr(0));
  // The reduction happens only in the penultimate dimension, which we now
  // tile.
  tileSizesSV[numLoops - 1] = rewriter.getIndexAttr(1);
  options = scf::SCFTilingOptions().setTileSizes(tileSizesSV);
  FailureOr<scf::SCFTilingResult> tileRes = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(splitRes->splitLinalgOp.getOperation()),
      options);
  if (failed(tileRes)) {
    LDBG() << "failed on step 3 (SCFTiling)";
    return failure();
  }
  rewriter.replaceOp(splitRes->splitLinalgOp, tileRes->replacements);
  return success();
}

/// Pass to splitReduce linalg operations.
class LLVMCPUSplitReductionPass
    : public impl::LLVMCPUSplitReductionPassBase<LLVMCPUSplitReductionPass> {
public:
  using Base::Base;
  explicit LLVMCPUSplitReductionPass(bool fpReductionReordering) {
    this->enableFpReductionReordering = fpReductionReordering;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUSplitReductionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  IRRewriter rewriter(context);
  SmallVector<linalg::GenericOp> candidates;
  funcOp.walk([&](linalg::GenericOp op) { candidates.push_back(op); });
  for (auto genericOp : candidates) {
    LDBG() << "candidate: " << genericOp;
    if (failed(splitReductionPrecondition(genericOp,
                                          enableFpReductionReordering))) {
      continue;
    }

    IREE::Codegen::LoweringConfigAttrInterface maybeLoweringConfig =
        getLoweringConfig(genericOp);
    if (!maybeLoweringConfig) {
      LDBG() << "can't find lowering_config, skip SplitReduction";
      continue;
    }
    auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        maybeLoweringConfig.getTilingLevelAttr(static_cast<unsigned>(
            IREE::CPU::TilingLevel::VectorReductionTiles)));
    ArrayRef<bool> scalableDims = attr.getScalableFlags();
    if (scalableDims.back()) {
      LDBG() << "scalable reduction dimensions not yet supported, skip "
                "SplitReduction";
      continue;
    }
    ArrayRef<int64_t> reductionSizes = attr.getSizes();
    if (reductionSizes.empty()) {
      LDBG()
          << "the list of reduction tiling sizes is empty, skip SplitReduction";
      continue;
    }
    int64_t size = reductionSizes.back();
    if (failed(splitReductionImpl(genericOp, size, rewriter))) {
      return signalPassFailure();
    }
  }
}
} // namespace
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUSplitReductionPass(const bool enableFpReductionReordering) {
  return std::make_unique<LLVMCPUSplitReductionPass>(
      enableFpReductionReordering);
}
} // namespace mlir::iree_compiler
