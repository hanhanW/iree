// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace {
struct DataLayoutPropagationPass
    : public DataLayoutPropagationBase<DataLayoutPropagationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    {
      RewritePatternSet patterns(context);
      linalg::populateDataLayoutPropagationPatterns(
          patterns, [](Operation *op) {
            // Do not propagate tensor.pack/unpack ops through reduction ops.
            if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
              return linalgOp.getNumReductionLoops() == 0;
            }
            return true;
          });
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns))))
        return signalPassFailure();
    }

    {
      RewritePatternSet patterns(context);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
      context->getLoadedDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns))))
        return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDataLayoutPropagationPass() {
  return std::make_unique<DataLayoutPropagationPass>();
}

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
