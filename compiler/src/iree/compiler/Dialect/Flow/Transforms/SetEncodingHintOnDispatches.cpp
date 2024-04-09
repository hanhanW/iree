// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <queue>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::Flow {

namespace {

struct SetEncodingHintOnDispatchesPass
    : public SetEncodingHintOnDispatchesBase<SetEncodingHintOnDispatchesPass> {
  void runOnOperation() override;
};

} // namespace

static LogicalResult lowerUpperBoundTileSizeOpToConstantsAndAttachEncodingHints(
    RewriterBase &rewriter,
    LinalgExt::UpperBoundTileSizeOp upperBoundTileSizeOp) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(upperBoundTileSizeOp);

  AffineExpr s0, s1;
  bindSymbols(rewriter.getContext(), s0, s1);
  AffineMap roundMap = AffineMap::get(0, 2, s1.ceilDiv(s0) * s0);

  SmallVector<affine::AffineApplyOp> applyOps;
  SmallVector<DispatchWorkgroupsOp> dispatches;
  std::queue<Operation *> que;
  que.push(upperBoundTileSizeOp);
  while (!que.empty()) {
    auto op = que.front();
    que.pop();
    for (Operation *user : op->getUsers()) {
      if (auto applyOp = dyn_cast<affine::AffineApplyOp>(user)) {
        if (applyOp.getMap() != roundMap) {
          return failure();
        }
        applyOps.push_back(applyOp);
        que.push(applyOp);
      } else if (auto dispatch = dyn_cast<DispatchWorkgroupsOp>(user)) {
        dispatches.push_back(dispatch);
      } else {
        return failure();
      }
    }
  }

  constexpr int64_t kAlignment = 16;
  Location loc = upperBoundTileSizeOp.getLoc();
  Value cst = rewriter.createOrFold<arith::ConstantIndexOp>(loc, kAlignment);
  for (auto value : upperBoundTileSizeOp.getResults()) {
    rewriter.replaceAllUsesWith(value, cst);
  }
  for (auto applyOp : applyOps) {
    rewriter.replaceAllUsesWith(applyOp.getResult(),
                                applyOp.getMapOperands().back());
  }

  auto encodingAttr = Codegen::EncodingRoundDimsToAttr::get(
      rewriter.getContext(), kAlignment);
  for (auto dispatch : dispatches) {
    dispatch->setAttr(Codegen::EncodingRoundDimsToAttr::getMnemonic(),
                      encodingAttr);
  }

  return success();
}
void SetEncodingHintOnDispatchesPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  Operation *funcOp = getOperation();
  IRRewriter rewriter(ctx);
  auto res =
      funcOp->walk([&](LinalgExt::UpperBoundTileSizeOp op) -> WalkResult {
        if (failed(lowerUpperBoundTileSizeOpToConstantsAndAttachEncodingHints(
                rewriter, op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createSetEncodingHintOnDispatchesPass() {
  return std::make_unique<SetEncodingHintOnDispatchesPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
