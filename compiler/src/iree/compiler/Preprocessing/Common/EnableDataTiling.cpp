// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace {

static tensor::PackOp createPackOp(RewriterBase &rewriter, Value src,
                                   ArrayRef<int64_t> innerTileSizes,
                                   ArrayRef<int64_t> innerDimsPos,
                                   ArrayRef<int64_t> outerDimsPerm) {
  auto srcType = src.getType().cast<RankedTensorType>();
  assert(srcType.getRank() == 2 && innerTileSizes.size() == 2 &&
         innerDimsPos.size() == 2);
  Location loc = src.getLoc();
  bool needPaddingValue = false;
  for (auto [pos, tileSize] : llvm::zip_equal(innerDimsPos, innerTileSizes)) {
    if (srcType.isDynamicDim(pos))
      needPaddingValue = true;
    else if (srcType.getDimSize(pos) % tileSize != 0)
      needPaddingValue = true;
  }

  Value paddingValue;
  if (needPaddingValue)
    paddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(srcType.getElementType()));

  SmallVector<OpFoldResult> ofr;
  for (auto i : innerTileSizes) ofr.push_back(rewriter.getI64IntegerAttr(i));
  auto dest = tensor::PackOp::createDestinationTensor(
      rewriter, loc, src, ofr, innerDimsPos, outerDimsPerm);
  return rewriter.create<tensor::PackOp>(loc, src, dest, innerDimsPos, ofr,
                                         paddingValue, outerDimsPerm);
}

static void rewriteMatmulOp(linalg::MatmulOp matmulOp, RewriterBase &rewriter) {
  auto lhs = matmulOp.getDpsInputOperand(0)->get();
  auto rhs = matmulOp.getDpsInputOperand(1)->get();
  auto init = matmulOp.getDpsInitOperand(0)->get();

  auto packedLhs = createPackOp(rewriter, lhs, {16, 1}, {0, 1}, {});
  auto packedRhs = createPackOp(rewriter, rhs, {16, 1}, {1, 0}, {1, 0});
  auto packedInit = createPackOp(rewriter, init, {16, 16}, {0, 1}, {});

  Location loc = matmulOp.getLoc();
  auto mmt4dOp = rewriter.create<linalg::Mmt4DOp>(
      loc, packedInit.getType(), ValueRange{packedLhs, packedRhs},
      ValueRange{packedInit});
  SmallVector<OpFoldResult> unpackInnerTiles(2, rewriter.getI64IntegerAttr(16));

  SmallVector<OpFoldResult> unpackEmptyShape;
  unpackEmptyShape.push_back(
      rewriter.createOrFold<tensor::DimOp>(loc, init, 0));
  unpackEmptyShape.push_back(
      rewriter.createOrFold<tensor::DimOp>(loc, init, 1));
  Value unpackEmpty = rewriter.create<tensor::EmptyOp>(
      loc, unpackEmptyShape, packedInit.getDestType().getElementType());
  if (unpackEmpty.getType() != init.getType()) {
    unpackEmpty =
        rewriter.create<tensor::CastOp>(loc, init.getType(), unpackEmpty);
  }
  auto unpackOp = rewriter.create<tensor::UnPackOp>(
      loc, mmt4dOp.getResult(0), unpackEmpty, ArrayRef<int64_t>{0, 1},
      unpackInnerTiles, ArrayRef<int64_t>{});
  rewriter.replaceOp(matmulOp, unpackOp.getResult());
}

struct EnableDataTilingPass : EnableDataTilingBase<EnableDataTilingPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    auto funcOp = getOperation();
    funcOp.walk([&](linalg::MatmulOp matmulOp) {
      rewriter.setInsertionPointAfter(matmulOp);
      rewriteMatmulOp(matmulOp, rewriter);
    });

    {
      RewritePatternSet patterns(context);
      context->getLoadedDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      tensor::CastOp::getCanonicalizationPatterns(patterns, context);
      tensor::EmptyOp::getCanonicalizationPatterns(patterns, context);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createEnableDataTilingPass() {
  return std::make_unique<EnableDataTilingPass>();
}

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
