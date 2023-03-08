// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
namespace {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
  public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
}  // namespace

class ConstEvalSetEncodingOpsPass
    : public ConstEvalSetEncodingOpsBase<ConstEvalSetEncodingOpsPass> {
 public:
  ConstEvalSetEncodingOpsPass() = default;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    SmallVector<IREE::LinalgExt::SetEncodingOp> candidates;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      funcOp.walk([&](IREE::LinalgExt::SetEncodingOp op) {
        if (isa<arith::ConstantOp>(op.getSource().getDefiningOp()))
          candidates.push_back(op);
      });
    }
    SimpleRewriter globalRewriter(&getContext());
    globalRewriter.setInsertionPointToEnd(moduleOp.getBody());
    auto initializerOp =
        globalRewriter.create<IREE::Util::InitializerOp>(moduleOp.getLoc());
    OpBuilder initBuilder =
        OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
    int64_t initCount = 0;
    globalRewriter.setInsertionPointToStart(moduleOp.getBody());
    for (auto op : candidates) {
      std::string globalOpName =
          ("_const_eval_set_encoding_" + llvm::Twine(initCount++)).str();

      Location loc = op.getLoc();
      auto global = globalRewriter.create<IREE::Util::GlobalOp>(
          loc, globalOpName, /*isMutable=*/false, op.getResultType());
      global.setPrivate();
      auto cst = initBuilder.clone(*op.getSource().getDefiningOp());
      auto setEncoding = initBuilder.create<IREE::LinalgExt::SetEncodingOp>(
          loc, cst->getResult(0), op.getResultTensorEncoding());
      initBuilder.create<IREE::Util::GlobalStoreOp>(
          loc, setEncoding.getResult(), global.getSymName());

      SimpleRewriter innerRewriter(&getContext());
      innerRewriter.setInsertionPoint(op);
      innerRewriter.replaceOpWithNewOp<IREE::Util::GlobalLoadOp>(op, global);
    }
    initBuilder.create<IREE::Util::InitializerReturnOp>(moduleOp.getLoc());
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConstEvalSetEncodingOpsPass() {
  return std::make_unique<ConstEvalSetEncodingOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
