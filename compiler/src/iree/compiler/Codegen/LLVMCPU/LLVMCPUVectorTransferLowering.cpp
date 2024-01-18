// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-vector-transfer-lowering"

namespace mlir::iree_compiler {
namespace {
struct BitCastLowering
    : public OpRewritePattern<vector::BitCastOp> {
  BitCastLowering(MLIRContext *context, unsigned maxRank,
                  PatternBenefit benefit = 1)
      : OpRewritePattern<vector::BitCastOp>(context, benefit),
        targetRank(maxRank) {}

  LogicalResult matchAndRewrite(vector::BitCastOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<VectorType>(op.getResult().getType());
    if (resultType.getRank() <= targetRank)
      return failure();

    Location loc = op.getLoc();
    Value res = rewriter.create<vector::SplatOp>(
        loc, resultType,
        rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(resultType.getElementType())));

    VectorType newResultType = VectorType::Builder(resultType).dropDim(0);

    int64_t dimSize = resultType.getShape()[0];
    for (int64_t i = 0; i < dimSize; ++i) {
      Value vec = rewriter.create<vector::ExtractOp>(loc, op.getSource(), i);
      vec = rewriter.create<vector::BitCastOp>(loc, newResultType, vec);
      res = rewriter.create<vector::InsertOp>(loc, vec, res, i);
    }

    rewriter.replaceOp(op, res);

    return success();
  }

  unsigned targetRank;
};

class LLVMCPUVectorTransferLoweringPass
    : public LLVMCPUVectorTransferLoweringBase<
          LLVMCPUVectorTransferLoweringPass> {
public:
  using LLVMCPUVectorTransferLoweringBase::LLVMCPUVectorTransferLoweringBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVectorTransferLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  {
    RewritePatternSet patterns(ctx);
    patterns.insert<BitCastLowering>(ctx, 1);
    vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
    vector::InsertOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  RewritePatternSet patterns(ctx);
  vector::populateVectorTransferLoweringPatterns(patterns,
                                                 /*maxTransferRank=*/1);
  auto vectorTransferToSCFOptions =
      VectorTransferToSCFOptions().enableFullUnroll();
  populateVectorToSCFConversionPatterns(patterns, vectorTransferToSCFOptions);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUVectorTransferLoweringPass() {
  return std::make_unique<LLVMCPUVectorTransferLoweringPass>();
}

} // namespace mlir::iree_compiler
