// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-virtual-vector-lowering"

namespace mlir::iree_compiler {
namespace {
class PackTransposeLastDim
    : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> perm = op.getPermutation();
    if (perm.back() + 1 != perm.size()) {
      return rewriter.notifyMatchFailure(
          op, "expect the last dim is not transposed");
    }
    VectorType srcType = op.getSourceVectorType();
    if (srcType.getShape().back() == 1) {
      return rewriter.notifyMatchFailure(
          op, "expect there are more than 1 elements in the last dimension");
    }

    Type elemType = srcType.getElementType();
    int64_t newElemBitwidth =
        elemType.getIntOrFloatBitWidth() * srcType.getShape().back();
    if (newElemBitwidth > 32) {
      return rewriter.notifyMatchFailure(
          op, "do not pack because the new bitwidth will be greater than 32");
    }

    Type newElemType = rewriter.getIntegerType(newElemBitwidth);
    SmallVector<int64_t> newSrcShape(srcType.getShape());
    newSrcShape.back() = 1;
    auto newSrcType = VectorType::get(newSrcShape, newElemType);

    VectorType resType = op.getResultVectorType();
    SmallVector<int64_t> newResShape(resType.getShape());
    newResShape.back() = 1;
    auto newResType = VectorType::get(newResShape, newElemType);

    Location loc = op.getLoc();
    auto srcBitCast =
        rewriter.create<vector::BitCastOp>(loc, newSrcType, op.getVector());
    auto transpose =
        rewriter.create<vector::TransposeOp>(loc, newResType, srcBitCast, perm);
    rewriter.replaceOpWithNewOp<vector::BitCastOp>(op, resType, transpose);
    return success();
  }
};


class LLVMCPUVirtualVectorLoweringPass
    : public LLVMCPUVirtualVectorLoweringBase<
          LLVMCPUVirtualVectorLoweringPass> {
public:
  using LLVMCPUVirtualVectorLoweringBase::LLVMCPUVirtualVectorLoweringBase;
  LLVMCPUVirtualVectorLoweringPass(std::string splitVectorTransfersTo) {
    this->splitVectorTransfersTo = splitVectorTransfersTo;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVirtualVectorLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  auto vectorMultiReductionLowering =
      vector::VectorMultiReductionLowering::InnerReduction;
  auto vectorContractLowering = vector::VectorContractLowering::OuterProduct;
  auto vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  auto vectorTransformOptions =
      vector::VectorTransformsOptions()
          .setVectorTransformsOptions(vectorContractLowering)
          .setVectorMultiReductionLowering(vectorMultiReductionLowering)
          .setVectorTransferSplit(vectorTransferSplit);

  RewritePatternSet patterns(ctx);
  {
    RewritePatternSet patterns(ctx);
    patterns.insert<PackTransposeLastDim>(ctx);
      vector::populateBubbleVectorBitCastOpPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  // TODO(hanchung): Maybe we should move drop unit dims patterns to a separate
  // pass. We are abusing OptimizeVectorTransferPass with `flatten=true` in some
  // CPU pipelines.
  vector::populateVectorTransferDropUnitDimsPatterns(patterns);
  vector::populateVectorTransferCollapseInnerMostContiguousDimsPatterns(patterns);
  vector::populateVectorGatherLoweringPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(
      patterns, vectorTransformOptions,
      /*benefit=*/1,
      /*disableOuterProductLowering=*/false);
  // This pattern will transform vector loads whose elements are used in a
  // scalar fashion into scalar loads. This will let scalar loads to be folded
  // into broadcast/arithmetic operations and reduce register pressure.
  vector::populateScalarVectorTransferLoweringPatterns(
      patterns, /*benefit=*/1, /*allowMultipleUses=*/true);
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vectorMultiReductionLowering);
  populateVectorTransferFullPartialPatterns(patterns, vectorTransformOptions);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUVirtualVectorLoweringPass(std::string splitVectorTransfersTo) {
  return std::make_unique<LLVMCPUVirtualVectorLoweringPass>(
      splitVectorTransfersTo);
}

} // namespace mlir::iree_compiler
