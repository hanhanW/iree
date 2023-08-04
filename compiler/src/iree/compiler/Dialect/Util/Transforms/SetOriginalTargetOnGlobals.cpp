// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {
namespace {
class SetOriginalTargetOnGlobalsPass
    : public SetOriginalTargetOnGlobalsBase<SetOriginalTargetOnGlobalsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Check to see if targets are already specified.
    auto targetsAttr = moduleOp->getAttrOfType<ArrayAttr>("hal.device.targets");
    if (!targetsAttr)
      return;

    // The encoding is associated with target platform. Different target could
    // materialize differently. Only support single target cases. If we want to
    // support multi-target, we will need to allocate different globals for
    // different targets.
    if (targetsAttr.size() != 1)
      return;

    moduleOp.walk([&](Util::InitializerOp op) {
      if (!op->getAttr("iree.compiler.consteval"))
        return WalkResult::advance();
      op->setAttr("iree.compiler.consteval.encoding.target", targetsAttr[0]);
      return WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createSetOriginalTargetOnGlobalsPass() {
  return std::make_unique<SetOriginalTargetOnGlobalsPass>();
}

} // namespace Util
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
