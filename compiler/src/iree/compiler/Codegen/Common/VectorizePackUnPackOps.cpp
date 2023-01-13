// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

struct VectorizePackUnPackOpsPass
    : public VectorizePackUnPackOpsBase<VectorizePackUnPackOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect,
                    arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    // Apply tiling to make outer dims be all 1s.
    {
      SimpleRewriter rewriter(ctx);
      auto packOptions = scf::SCFTileAndFuseOptions().setTilingOptions(
          scf::SCFTilingOptions().setTileSizeComputationFunction(
              [](OpBuilder &builder, Operation *op) -> SmallVector<Value> {
                Location loc = op->getLoc();
                auto packOp = cast<tensor::PackOp>(op);

                // Do nothing if any of inner tile sizes is dynamic.
                if (llvm::any_of(packOp.getMixedTiles(), [](OpFoldResult tile) {
                      return tile.is<Value>();
                    }))
                  return {};

                int inputRank = packOp.getSourceRank();
                SmallVector<Value> tileSizes(
                    inputRank, builder.create<arith::ConstantIndexOp>(loc, 1));
                return tileSizes;
              }));
      auto funcOp = getOperation();
      funcOp->walk([&](tensor::PackOp op) {
        FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
            scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
                rewriter, cast<TilingInterface>(op.getOperation()),
                packOptions);
        if (failed(tileAndFuseResult)) return signalPassFailure();
        rewriter.replaceOp(op,
                           tileAndFuseResult->replacements[op.getResult()]);
      });

      auto unpackTilingOptions =
          scf::SCFTilingOptions().setTileSizeComputationFunction(
              [](OpBuilder &builder, Operation *op) {
                Location loc = op->getLoc();
                auto unpackOp = cast<tensor::UnPackOp>(op);
                int numLoops = unpackOp.getDestRank();
                auto dimAndTileMapping = unpackOp.getDimAndTileMapping();
                SmallVector<Value> tileSizes;
                for (int i = 0; i < numLoops; ++i) {
                  if (dimAndTileMapping.count(i)) {
                    tileSizes.push_back(getValueOrCreateConstantIndexOp(
                        builder, loc, dimAndTileMapping[i]));
                  } else {
                    tileSizes.push_back(builder.createOrFold<tensor::DimOp>(
                        loc, unpackOp.getDest(), i));
                  }
                }
                return tileSizes;
              });
      funcOp->walk([&](tensor::UnPackOp op) {
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForOp(
            rewriter, cast<TilingInterface>(op.getOperation()),
            unpackTilingOptions);
        if (failed(tilingResult)) return signalPassFailure();
        rewriter.replaceOp(op, tilingResult->replacements);
      });
    }

    // Generalize pack and unpack ops and canonicalize tiled ops.
    {
      RewritePatternSet patterns(ctx);
      linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
      patterns.add<linalg::GeneralizeOuterUnitDimsPackOpPattern,
                   linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(ctx);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Kick in generic vectorizer.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<IREE::LinalgExt::LinalgVectorizationPattern>(ctx);
      linalg::populatePadOpVectorizationPatterns(patterns);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
      vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
      // TODO(hanchung): Capture the failure after the vectorization pattern
      // rewrite converges.
      (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)));
    }
  }
};
}  // namespace
std::unique_ptr<OperationPass<func::FuncOp>>
createVectorizePackUnPackOpsPass() {
  return std::make_unique<VectorizePackUnPackOpsPass>();
}
}  // namespace iree_compiler
}  // namespace mlir
