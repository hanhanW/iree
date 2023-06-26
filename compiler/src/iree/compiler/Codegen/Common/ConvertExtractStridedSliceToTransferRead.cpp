
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hoist-redundant-vector-transfers"
#define VEC_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace iree_compiler {
namespace {

struct HanhanPattern : public OpRewritePattern<vector::ExtractStridedSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const final {
    auto readOp = op.getVector().getDefiningOp<vector::TransferReadOp>();
    if (!readOp) return failure();
    SmallVector<OpFoldResult> indices = readOp.getIndices();
    SmallVector<int64_t> offsets;
    SmallVector<OpFoldResult> newIndices;
    op.getOffsets(offsets);
    auto loc = op.getLoc();
    for (const auto [idx, offset] : llvm::zip_equal(indices, offsets)) {
      AffineExpr s0 = getAffineSymbolExpr(0, rewriter.getContext());
      newIndices.push_back(affine::makeComposedFoldedAffineApply(
          rewriter, loc, s0 + offset, idx));
    }
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        op, op.getType(), readOp.getSource(),
        getValueOrCreateConstantIndexOp(rewriter, loc, newIndices),
        readOp.getPermutationMapAttr(), readOp.getPadding(), readOp.getMask(),
        readOp.getInBoundsAttr());
    return success();
  }
};

struct HanhanPattern2 : public OpRewritePattern<vector::ExtractStridedSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const final {
    auto cstOp = op.getVector().getDefiningOp<arith::ConstantOp>();
    if (!cstOp) return failure();
    auto splatAttr = llvm::dyn_cast<SplatElementsAttr>(cstOp.getValue());
    if (!splatAttr || !splatAttr.isSplat()) return failure();
    auto vtType = llvm::cast<ShapedType>(op.getType());
    auto newAttr = splatAttr.resizeSplat(vtType);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newAttr);
    return success();
  }
};



class ConvertExtractStridedSliceToTransferReadPass
    : public ConvertExtractStridedSliceToTransferReadBase<
          ConvertExtractStridedSliceToTransferReadPass> {
 public:
  using ConvertExtractStridedSliceToTransferReadBase::
      ConvertExtractStridedSliceToTransferReadBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void ConvertExtractStridedSliceToTransferReadPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<HanhanPattern, HanhanPattern2>(ctx);
  auto funcOp = getOperation();
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertExtractStridedSliceToTransferReadPass() {
  return std::make_unique<ConvertExtractStridedSliceToTransferReadPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
