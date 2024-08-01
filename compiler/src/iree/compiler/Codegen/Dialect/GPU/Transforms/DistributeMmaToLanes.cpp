// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_DISTRIBUTEMMATOLANESPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct DistributeMmaToLanesPass final
    : impl::DistributeMmaToLanesPassBase<DistributeMmaToLanesPass> {
  void runOnOperation() override;
};
} // namespace

struct ConvertToMultiMma final : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    auto loweringConfig =
        getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
    if (!loweringConfig) {
      return failure();
    }
    IREE::GPU::MmaInterfaceAttr kind = loweringConfig.getMmaKind();
    if (!kind) {
      return failure();
    }
    if (failed(convertContractionToMultiMma(rewriter, linalgOp, kind))) {
      return failure();
    }
    return success();
  }
};

LogicalResult fuseProducersGreedily(RewriterBase &rewriter,
                                    scf::ForallOp laneForall) {

  std::deque<tensor::ExtractSliceOp> candidates;
  laneForall->walk([&](tensor::ExtractSliceOp extractSliceOp) {
    auto producer = extractSliceOp.getSource().getDefiningOp<TilingInterface>();
    if (producer && producer->getBlock() != laneForall.getBody()) {
      candidates.push_back(extractSliceOp);
    }
  });

  SmallVector<LoopLikeOpInterface> loops = {laneForall};

  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    // Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();

    // Materialize the slice of the producer in place.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
        scf::tileAndFuseProducerOfSlice(rewriter, candidateSliceOp, loops);
    if (!fusedProducer)
      continue;

    // We have no way to know whether a multi-use value can be yielded from the
    // parallel loop so never yield a replacement.

    // Add more fusion candidates to the worklist.
    for (auto tiledOp : fusedProducer->tiledOps) {
      for (OpOperand &operand : tiledOp->getOpOperands()) {
        auto sliceOp = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
        if (!sliceOp)
          continue;
        candidates.push_back(sliceOp);
      }
    }
  }
  return success();
}

void DistributeMmaToLanesPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  // Step 1. Convert configured linalg ops to multi_mma.
  {
    RewritePatternSet patterns(context);
    patterns.add<ConvertToMultiMma>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "failed to convert linalg to multi_mma";
      return signalPassFailure();
    }
  }
  funcOp.dump();

  // Step 2. Distribute multi_mma ops to lanes and greedily fuse producers.
  SmallVector<IREE::GPU::MultiMmaOp> mmaOps;
  funcOp.walk([&](IREE::GPU::MultiMmaOp mmaOp) { mmaOps.push_back(mmaOp); });
  IRRewriter rewriter(funcOp);
  for (auto mmaOp : mmaOps) {
    rewriter.setInsertionPoint(mmaOp);
    FailureOr<scf::ForallOp> maybeLaneForall =
        distributeMultiMmaOp(rewriter, mmaOp);
    if (failed(maybeLaneForall)) {
      funcOp.emitError() << "failed to distribute multi_mma ops to lanes";
      return signalPassFailure();
    }

    rewriter.setInsertionPointToStart(maybeLaneForall->getBody());
    if (failed(fuseProducersGreedily(rewriter, *maybeLaneForall))) {
      funcOp.emitError() << "failed to fuse producers into lane forall";
      return signalPassFailure();
    }
  }

  // Post distribution cleanup patterns.
  {
    RewritePatternSet patterns(context);
    // Merge consecutive insert/extract slice ops to simplify later loop
    // hoisting patterns.
    tensor::populateFoldTensorEmptyPatterns(patterns);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "cleanup failed\n";
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::IREE::GPU
