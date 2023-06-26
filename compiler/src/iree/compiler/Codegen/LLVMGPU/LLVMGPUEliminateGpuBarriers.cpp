// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hoist-redundant-vector-transfers"
#define VEC_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace iree_compiler {
namespace {

class EliminateGpuBarriersPass
    : public LLVMGPUEliminateGpuBarriersBase<
          EliminateGpuBarriersPass> {
 public:
  using LLVMGPUEliminateGpuBarriersBase::LLVMGPUEliminateGpuBarriersBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }
  void runOnOperation() override;
};

void EliminateGpuBarriersPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  auto funcOp = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<BarrierElimination>(ctx);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUEliminateGpuBarriersPass() {
  return std::make_unique<EliminateGpuBarriersPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
