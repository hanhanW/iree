// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-vector-transpose-lowering"

namespace mlir::iree_compiler {
namespace {

static bool areDimsTransposedIn2DSlice(int64_t dim0, int64_t dim1,
                                       ArrayRef<int64_t> transp) {
  // Perform a linear scan along the dimensions of the transposed pattern. If
  // dim0 is found first, dim0 and dim1 are not transposed within the context of
  // their 2D slice. Otherwise, 'dim1' is found first and they are transposed.
  for (int64_t permDim : transp) {
    if (permDim == dim0)
      return false;
    if (permDim == dim1)
      return true;
  }

  llvm_unreachable("Ill-formed transpose pattern");
}

static FailureOr<SmallVector<int64_t>>
isTarget3DTransposeSlice(vector::TransposeOp op) {
  VectorType srcType = op.getSourceVectorType();
  SmallVector<int64_t> srcGtOneDims;
  for (auto [index, size] : llvm::enumerate(srcType.getShape()))
    if (size > 1)
      srcGtOneDims.push_back(index);

  if (srcGtOneDims.size() != 2 && srcGtOneDims.size() != 3)
    return failure();

  // Check whether the two source vector dimensions that are greater than one
  // must be transposed with each other so that we can apply one of the 2-D
  // transpose pattens. Otherwise, these patterns are not applicable.
  if (!areDimsTransposedIn2DSlice(srcGtOneDims[0], srcGtOneDims[1],
                                  op.getPermutation()))
    return failure();

  return srcGtOneDims;
}

static bool has16x16Transpose(mlir::FunctionOpInterface funcOp) {
  bool res = false;
  funcOp.walk([&](vector::TransposeOp op) {
    auto srcGtOneDims = isTarget3DTransposeSlice(op);
    if (failed(srcGtOneDims))
      return WalkResult::advance();
    VectorType srcType = op.getSourceVectorType();
    int64_t m = srcType.getDimSize(srcGtOneDims.value()[0]);
    int64_t n = srcType.getDimSize(srcGtOneDims.value()[1]);
    if (m == 16 && n == 16) {
      res = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return res;
}
class AVX512HanhanLowering : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSourceVectorType().isScalable())
      return rewriter.notifyMatchFailure(
          op, "vector shuffle lowering not supported for scalable vectors");
    if (op.getPermutation().back() != op.getPermutation().size() - 1) {
      return rewriter.notifyMatchFailure(
          op, "expected inner most is not transposed");
    }
    auto srcGtOneDims = isTarget3DTransposeSlice(op);
    if (failed(srcGtOneDims))
      return rewriter.notifyMatchFailure(
          op, "expected transposition on a 2D slice");
    auto shapeCast = op.getVector().getDefiningOp<vector::ShapeCastOp>();
    if (!shapeCast)
      return failure();
    VectorType srcType = shapeCast.getSourceVectorType();
    VectorType resType = shapeCast.getResultVectorType();
    int srcRank = srcType.getRank();
    int resRank = resType.getRank();
    if (resRank < 2 || srcRank > resRank)
      return failure();
    if (resType.getShape()[resRank - 2] != 16 ||
        resType.getShape()[resRank - 1] != 2)
      return failure();
    if (srcType.getShape()[srcRank - 1] != 32)
      return failure();
    if (!srcType.getElementType().isBF16())
      return failure();
    SmallVector<int64_t> newSrcShape(srcType.getShape());
    newSrcShape.back() /= 2;
    VectorType newSrcType = VectorType::get(newSrcShape, rewriter.getF32Type());

    Location loc = op.getLoc();
    auto bitcast = rewriter.create<vector::BitCastOp>(loc, newSrcType,
                                                      shapeCast.getSource());

    SmallVector<int64_t> newShapeCastShape(resType.getShape());
    newShapeCastShape.pop_back();
    VectorType newShapeCastType =
        VectorType::get(newShapeCastShape, rewriter.getF32Type());

    auto newShapeCast =
        rewriter.create<vector::ShapeCastOp>(loc, newShapeCastType, bitcast);

    SmallVector<int64_t> perm(op.getPermutation().drop_back());
    auto transpose =
        rewriter.create<vector::TransposeOp>(loc, newShapeCast, perm);

    SmallVector<int64_t> newResShape(resType.getShape());
    newResShape[resRank - 2] *= newResShape[resRank - 1];
    newResShape.pop_back();
    VectorType newResType =
        VectorType::get(newResShape, rewriter.getBF16Type());
    auto bitcast2 =
        rewriter.create<vector::BitCastOp>(loc, newResType, transpose);

    auto shapecast2 =
        rewriter.create<vector::ShapeCastOp>(loc, resType, bitcast2);
    rewriter.replaceOp(op, shapecast2);
    return success();
  }
};

class UnrollBitCastOp final : public OpRewritePattern<vector::BitCastOp> {
public:
  UnrollBitCastOp(int64_t targetRank, MLIRContext *context,
                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), targetRank(targetRank){};

  LogicalResult matchAndRewrite(vector::BitCastOp op,
                                PatternRewriter &rewriter) const override {
    VectorType resultType = op.getResultVectorType();
    auto unrollIterator = vector::createUnrollIterator(resultType, targetRank);
    if (!unrollIterator)
      return failure();

    auto loc = op.getLoc();
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultType, rewriter.getZeroAttr(resultType));
    SmallVector<int64_t> shape = {
        resultType.getShape()[resultType.getRank() - 1]};

    auto bitcastResType = VectorType::get(shape, resultType.getElementType());
    for (auto position : *unrollIterator) {
      Value extract =
          rewriter.create<vector::ExtractOp>(loc, op.getSource(), position);
      Value bitcast =
          rewriter.create<vector::BitCastOp>(loc, bitcastResType, extract);
      result =
          rewriter.create<vector::InsertOp>(loc, bitcast, result, position);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  int64_t targetRank = 1;
};

class LLVMCPUVectorTransposeLoweringPass
    : public LLVMCPUVectorTransposeLoweringBase<
          LLVMCPUVectorTransposeLoweringPass> {
public:
  using LLVMCPUVectorTransposeLoweringBase::LLVMCPUVectorTransposeLoweringBase;
  LLVMCPUVectorTransposeLoweringPass(bool lowerVectorTransposeToAVX2) {
    this->lowerVectorTransposeToAVX2 = lowerVectorTransposeToAVX2;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVectorTransposeLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  auto vectorTransformOptions =
      vector::VectorTransformsOptions().setVectorTransposeLowering(
          vector::VectorTransposeLowering::Shuffle1D);
  if (has16x16Transpose(funcOp)) {
    vectorTransformOptions.setVectorTransposeLowering(
        vector::VectorTransposeLowering::Shuffle16x16);
  }

  RewritePatternSet patterns(ctx);
  patterns.add<AVX512HanhanLowering>(ctx);
  patterns.add<UnrollBitCastOp>(1, ctx);

  constexpr unsigned kSpecializedBenefit = 10;
  constexpr unsigned kNarrowTypeEmulationBenefit = 20;

  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(patterns,
                                                  vectorTransformOptions);
  vector::populateVectorTransposeNarrowTypeRewritePatterns(
      patterns, kNarrowTypeEmulationBenefit);

  if (lowerVectorTransposeToAVX2) {
    auto avx2LoweringOptions =
        x86vector::avx2::LoweringOptions().setTransposeOptions(
            x86vector::avx2::TransposeLoweringOptions()
                .lower4x8xf32()
                .lower8x8xf32());
    x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
        patterns, avx2LoweringOptions, kSpecializedBenefit);
  }
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUVectorTransposeLoweringPass(bool lowerVectorTransposeToAVX2) {
  return std::make_unique<LLVMCPUVectorTransposeLoweringPass>(
      lowerVectorTransposeToAVX2);
}

} // namespace mlir::iree_compiler
