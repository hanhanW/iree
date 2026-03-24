// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-codegen-extract-workgroup-count-as-func"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_EXTRACTWORKGROUPCOUNTASFUNCPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct ExtractWorkgroupCountAsFuncPass final
    : impl::ExtractWorkgroupCountAsFuncPassBase<
          ExtractWorkgroupCountAsFuncPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Collect export ops first to avoid modifying module while walking.
    SmallVector<IREE::HAL::ExecutableExportOp> exportOps;
    moduleOp.walk([&](IREE::HAL::ExecutableExportOp exportOp) {
      if (!exportOp.getWorkgroupCount().empty()) {
        exportOps.push_back(exportOp);
      }
    });

    for (auto exportOp : exportOps) {
      if (failed(extractCountFunc(moduleOp, exportOp))) {
        return signalPassFailure();
      }
    }
  }

  LogicalResult extractCountFunc(ModuleOp moduleOp,
                                 IREE::HAL::ExecutableExportOp exportOp) {
    Region &countRegion = exportOp.getWorkgroupCount();
    if (countRegion.empty()) {
      return success();
    }

    Block &countBlock = countRegion.front();
    Location loc = exportOp.getLoc();
    MLIRContext *ctx = &getContext();

    // The count region signature is:
    //   (%device: !hal.device, %workload0: index, %workload1: index, ...)
    //   -> (index, index, index)
    // We drop the !hal.device argument and keep only index args.
    SmallVector<Type> inputTypes;
    for (unsigned i = 1; i < countBlock.getNumArguments(); ++i) {
      inputTypes.push_back(countBlock.getArgument(i).getType());
    }
    SmallVector<Type> resultTypes(3, IndexType::get(ctx));
    auto funcType = FunctionType::get(ctx, inputTypes, resultTypes);

    // Name: <export_name>_workgroup_count
    std::string funcName =
        (exportOp.getSymName() + "_workgroup_count").str();

    OpBuilder moduleBuilder(ctx);
    moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
    auto funcOp = func::FuncOp::create(moduleBuilder, loc, funcName, funcType);
    funcOp.setPublic();

    // Copy workgroup_size and subgroup_size attributes from the export op
    // onto the extracted function. These attributes are lost when
    // serialization removes the hal.executable.export op, but they are
    // needed by downstream tools (e.g., iree-device-codegen --output-dir).
    if (auto wgSize = exportOp.getWorkgroupSize()) {
      funcOp->setAttr("workgroup_size", *wgSize);
    }
    if (auto sgSize = exportOp.getSubgroupSize()) {
      funcOp->setAttr("subgroup_size",
                       IntegerAttr::get(IndexType::get(ctx), *sgSize));
    }

    // Clone the count region body into the new function.
    Block *entryBlock = funcOp.addEntryBlock();
    IRMapping mapper;

    // Map count region args to func args (skip !hal.device at index 0).
    for (unsigned i = 1; i < countBlock.getNumArguments(); ++i) {
      mapper.map(countBlock.getArgument(i),
                 entryBlock->getArgument(i - 1));
    }

    OpBuilder funcBuilder = OpBuilder::atBlockBegin(entryBlock);
    for (auto &op : countBlock.without_terminator()) {
      funcBuilder.clone(op, mapper);
    }

    // Replace hal.return with func.return.
    auto halReturn = cast<IREE::HAL::ReturnOp>(countBlock.getTerminator());
    SmallVector<Value> returnValues;
    for (Value operand : halReturn.getOperands()) {
      returnValues.push_back(mapper.lookupOrDefault(operand));
    }
    func::ReturnOp::create(funcBuilder, loc, returnValues);

    LLVM_DEBUG(llvm::dbgs() << "Extracted workgroup count function: "
                            << funcName << " with " << inputTypes.size()
                            << " args\n");
    return success();
  }
};

} // namespace
} // namespace mlir::iree_compiler
