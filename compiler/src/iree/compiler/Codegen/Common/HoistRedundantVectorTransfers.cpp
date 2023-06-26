// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-hoist-redundant-vector-transfers"
#define VEC_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace iree_compiler {
namespace {

class HoistRedundantVectorTransfersPass
    : public HoistRedundantVectorTransfersBase<
          HoistRedundantVectorTransfersPass> {
 public:
  using HoistRedundantVectorTransfersBase::HoistRedundantVectorTransfersBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void HoistRedundantVectorTransfersPass::runOnOperation() {
  auto funcOp = getOperation();
  linalg::hoistRedundantVectorTransfers(funcOp);
  linalg::hoistRedundantVectorTransfersOnTensor(funcOp);
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createHoistRedundantVectorTransfersPass() {
  return std::make_unique<HoistRedundantVectorTransfersPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
