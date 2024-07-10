// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {
struct LowerGenericOpsToCallsPass
    : LowerGenericOpsToCallsBase<LowerGenericOpsToCallsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, func::FuncDialect>();
  }
  void runOnOperation() override;
};
} // namespace

void LowerGenericOpsToCallsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto moduleOp = getOperation();
  FunctionOpInterface mainFunc;
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    auto tInfo = getTranslationInfo(funcOp);
    if (tInfo.getPassPipeline().getValue() ==
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMLIRUkernel) {
      mainFunc = funcOp;
      break;
    }
  }

  if (!mainFunc)
    return;

  if (!getTranslationInfo(mainFunc).getConfiguration().contains(
          "ukernel_entry"))
    return;

  StringRef ukernelFuncName =
      cast<StringAttr>(
          getTranslationInfo(mainFunc).getConfiguration().get("ukernel_entry"))
          .getValue();
  FunctionOpInterface ukernelFunc;
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    if (funcOp.getName() == ukernelFuncName) {
      ukernelFunc = funcOp;
      break;
    }
  }

  DictionaryAttr config = getTranslationInfo(ukernelFunc).getConfiguration();
  assert(config);
  SmallVector<AffineMap> indexingMaps = llvm::map_to_vector(
      config.getAs<ArrayAttr>("indexing_maps").getValue(),
      [&](Attribute attr) { return cast<AffineMapAttr>(attr).getAffineMap(); });
  SmallVector<int64_t> loopRanges(
      config.getAs<DenseI64ArrayAttr>("loop_range").asArrayRef());

  SmallVector<linalg::GenericOp> genericOps;
  mainFunc.walk([&](linalg::GenericOp op) { genericOps.push_back(op); });

  IRRewriter rewriter(ctx);
  for (auto genericOp : genericOps) {
    if (!genericOp.hasPureBufferSemantics()) {
      continue;
    }
    SmallVector<AffineMap> genericIndexingMaps =
        genericOp.getIndexingMapsArray();
    if (genericIndexingMaps != indexingMaps) {
      continue;
    }
    rewriter.setInsertionPointAfter(genericOp);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        genericOp, cast<func::FuncOp>(ukernelFunc), genericOp.getOperands());
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createLowerGenericOpsToCallsPass() {
  return std::make_unique<LowerGenericOpsToCallsPass>();
}

} // namespace mlir::iree_compiler
