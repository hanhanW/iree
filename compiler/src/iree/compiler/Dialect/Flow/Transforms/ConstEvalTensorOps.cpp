// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-const-eval-set-encoding-ops"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class ConstEvalTensorOpsPass
    : public ConstEvalTensorOpsBase<ConstEvalTensorOpsPass> {
 public:
  ConstEvalTensorOpsPass() = default;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    SmallVector<tensor::PackOp> candidates;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      funcOp.walk([&](tensor::PackOp op) {
        if (isa<arith::ConstantOp>(op.getSource().getDefiningOp()))
          candidates.push_back(op);
      });
    }
    IRRewriter globalRewriter(&getContext());
    globalRewriter.setInsertionPointToEnd(moduleOp.getBody());
    auto initializerOp =
        globalRewriter.create<IREE::Util::InitializerOp>(moduleOp.getLoc());
    OpBuilder initBuilder =
        OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
    int64_t initCount = 0;
    globalRewriter.setInsertionPointToStart(moduleOp.getBody());
    for (auto op : candidates) {
      std::string globalOpName =
          ("_const_eval_pack_" + llvm::Twine(initCount++)).str();

      Location loc = op->getLoc();
      auto global = globalRewriter.create<IREE::Util::GlobalOp>(
          loc, globalOpName, /*isMutable=*/false, op.getDestType());
      global.setPrivate();
      auto cst = initBuilder.clone(*op.getSource().getDefiningOp());
      auto dest = initBuilder.clone(*op.getDest().getDefiningOp());
      Value paddingValue = op.getPaddingValue();
      if (paddingValue)
        paddingValue =
            initBuilder.clone(*paddingValue.getDefiningOp())->getResult(0);
      auto packOp = initBuilder.create<tensor::PackOp>(
          loc, cst->getResult(0), dest->getResult(0), op.getInnerDimsPos(),
          op.getMixedTiles(), paddingValue, op.getOuterDimsPerm());
      initBuilder.create<IREE::Util::GlobalStoreOp>(
          loc, packOp.getResult(), global.getSymName());

      IRRewriter innerRewriter(&getContext());
      innerRewriter.setInsertionPoint(op);
      innerRewriter.replaceOpWithNewOp<IREE::Util::GlobalLoadOp>(op, global);
    }
    initBuilder.create<IREE::Util::InitializerReturnOp>(moduleOp.getLoc());
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConstEvalTensorOpsPass() {
  return std::make_unique<ConstEvalTensorOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
