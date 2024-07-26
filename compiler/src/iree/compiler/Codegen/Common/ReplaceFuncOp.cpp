// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iterator>
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-replace-func-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

llvm::cl::opt<std::string> clCodegenMLIRUkernelFileName(
    "iree-codegen-mlir-ukernel-file-name",
    llvm::cl::desc(
        "File path to a module containing a set of MLIR based ukernels."),
    llvm::cl::init(""));

namespace {

static LogicalResult getMatchedUkernels(StringRef fileName,
                                        FunctionOpInterface funcOp,
                                        FunctionOpInterface &ukernelOp) {
  std::optional<ModuleOp> ukernelModule;
  auto dialect = funcOp->getContext()
                     ->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
  auto maybeUkernelModule =
      dialect->getOrLoadUkernelModule(std::string(fileName));
  if (failed(maybeUkernelModule)) {
    funcOp.emitError() << "failed to load transform ukernel module: "
                       << fileName;
    return failure();
  }
  ukernelModule = *maybeUkernelModule;
  LDBG("--found ukernel library @" << fileName);

  for (auto candidate : ukernelModule->getOps<FunctionOpInterface>()) {
    if (funcOp.getName() == candidate.getName()) {
      ukernelOp = candidate;
      break;
    }
  }
  return success();
}

struct ReplaceFuncOpPass
    : public ReplaceFuncOpBase<ReplaceFuncOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registerTransformDialectTranslationDependentDialects(registry);
  }

  void runOnOperation() override {
    if (clCodegenMLIRUkernelFileName.empty()) {
      return;
    }
    auto moduleOp = getOperation();
    StringRef fileName = llvm::StringRef(clCodegenMLIRUkernelFileName);
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      FunctionOpInterface ukernelOp;
      if (failed(getMatchedUkernels(fileName, funcOp, ukernelOp))) {
        funcOp.emitError() << "failed to parse ukernel file: "
                           << clCodegenMLIRUkernelFileName;
        return signalPassFailure();
      }

      if (!ukernelOp) {
        LDBG("--did not find matching funcOp" << funcOp.getName());
        continue;
      }

      funcOp.getCallableRegion()->takeBody(*ukernelOp.getCallableRegion());
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createReplaceFuncOpPass() {
  return std::make_unique<ReplaceFuncOpPass>();
}

} // namespace mlir::iree_compiler
