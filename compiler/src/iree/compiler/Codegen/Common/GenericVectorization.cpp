// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-generic-vectorization"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GENERICVECTORIZATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
// Returns the vector sizes from the local lowering config or try to infer them
// from the tensor shapes and tiled loops in the IR.
static std::optional<SizesAndScalableFlags>
getVectorSizes(Operation *op, bool useConfiguredVectorSizes) {
  // Get vector sizes from the lowering config, if available in the op itself.
  std::unique_ptr<TilingConfig> tilingConfig =
      TilingConfig::create(getLoweringConfig(op));
  if (useConfiguredVectorSizes && tilingConfig) {
    LDBG() << "Use configured vector sizes from lowering config";
    auto [vectorSizes, scalableFlags] = tilingConfig->getVectorTileSizes();
    // Replace zeros in canonical vector shape to turn it into a valid shape.
    std::replace(vectorSizes.begin(), vectorSizes.end(), 0, 1);
    return std::make_pair(vectorSizes, scalableFlags);
  }

  // Try to infer the vector sizes from the IR.
  std::optional<SmallVector<int64_t>> vectorSizes;
  SmallVector<bool> scalableFlags;
  TypeSwitch<Operation *, void>(op)
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
        std::optional<VectorizationTileSizes> result =
            inferSizesFromIR(linalgOp, /*opResult=*/std::nullopt);
        if (result) {
          vectorSizes = result->vectorSizes;
          scalableFlags = result->vectorScalableFlags;
        }
      })
      .Case<linalg::PackOp, linalg::UnPackOp>([&](auto op) {
        std::optional<VectorizationTileSizes> result = inferSizesFromIR(op);
        if (result) {
          vectorSizes = result->vectorSizes;
        }
      })
      .Case<tensor::PadOp>([&](tensor::PadOp padOp) {
        auto ty = padOp.getResultType();
        // TODO(hanchung): Infer the vector sizes for pad op after
        // maskedVectorize method allows dynamic result shapes.
        if (!ty.hasStaticShape())
          return;
        vectorSizes = SmallVector<int64_t>(ty.getShape());
      })
      .Case<IREE::LinalgExt::GatherOp>([&](IREE::LinalgExt::GatherOp gatherOp) {
        std::optional<VectorizationTileSizes> result =
            inferSizesFromIR(gatherOp.getOutput());
        if (result) {
          vectorSizes = result->vectorSizes;
        }
      })
      .Default([&](Operation *) {});

  if (vectorSizes) {
    scalableFlags.resize(vectorSizes->size(), false);
    return std::make_pair(vectorSizes.value(), scalableFlags);
  }
  return std::nullopt;
}

static LogicalResult isWithinVectorSizeLimit(linalg::LinalgOp linalgOp,
                                             int64_t maxVectorSize) {
  int64_t maxFlatVecSize = 1;
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    auto type = llvm::dyn_cast<ShapedType>(operand.get().getType());
    if (!type)
      continue;
    if (!type.hasStaticShape())
      return failure();
    maxFlatVecSize = std::max(maxFlatVecSize, type.getNumElements());
  }
  return success(maxFlatVecSize < maxVectorSize);
}

class GenericVectorizationPass final
    : public impl::GenericVectorizationPassBase<GenericVectorizationPass> {
public:
  using impl::GenericVectorizationPassBase<
      GenericVectorizationPass>::GenericVectorizationPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tensor::TensorDialect, linalg::LinalgDialect,
                vector::VectorDialect, IREE::VectorExt::IREEVectorExtDialect>();
  }
  void runOnOperation() override;
};

void GenericVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  IRRewriter rewriter(context);
  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp>(op)) {
      if (isa<linalg::CopyOp>(op) && !vectorizeCopies) {
        return;
      }
      candidates.push_back(op);
    } else if (vectorizePadding && enableVectorMasking &&
               isa<tensor::PadOp>(op)) {
      candidates.push_back(op);
    } else if (enableVectorMasking &&
               isa<linalg::PackOp, linalg::UnPackOp>(op)) {
      candidates.push_back(op);
    } else if (isa<IREE::LinalgExt::GatherOp>(op)) {
      candidates.push_back(op);
    }
  });

  // The vector input sizes inference needs to use producers, so we apply
  // vectorization from bottom to top.
  std::reverse(candidates.begin(), candidates.end());
  for (Operation *op : candidates) {
    SmallVector<int64_t> vectorSizes;
    SmallVector<bool> scalableVecDims;
    if (enableVectorMasking) {
      std::optional<SizesAndScalableFlags> vectorSizesAndScalableDims =
          getVectorSizes(op, useConfiguredVectorSizes);
      if (vectorSizesAndScalableDims) {
        std::tie(vectorSizes, scalableVecDims) = *vectorSizesAndScalableDims;
      }
    }

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Do not vectorize the op if the vector size is greater than or equal
      // to limit.
      if (enableVectorMasking) {
        if (std::accumulate(vectorSizes.begin(), vectorSizes.end(), 1,
                            std::multiplies<int64_t>()) >= maxVectorSize)
          continue;
      } else {
        if (failed(isWithinVectorSizeLimit(linalgOp, maxVectorSize)))
          continue;
      }
    }
    // Pad scalable dims with `false` to match the vector sizes.
    scalableVecDims.resize(vectorSizes.size());

    // Try to vectorize to transfer_gather, if possible.
    if (isa<linalg::GenericOp>(op) && vectorizeToTransferGather) {
      (void)IREE::VectorExt::vectorizeGatherLikeGenericToTransferGather(
          rewriter, cast<linalg::GenericOp>(op), vectorSizes, scalableVecDims,
          vectorizeGatherAccesses);
    } else if (auto gatherOp = dyn_cast<IREE::LinalgExt::GatherOp>(op)) {
      (void)IREE::VectorExt::vectorizeLinalgExtGatherToTransferGather(
          rewriter, gatherOp, vectorSizes);
    } else {
      FailureOr<linalg::VectorizationResult> result = linalg::vectorize(
          rewriter, op, vectorSizes, scalableVecDims, vectorizeGatherAccesses);
      if (succeeded(result)) {
        rewriter.replaceOp(op, result->replacements);
      }
    }
  };

  {
    // Eliminate (all-true) vector masks as early as possible (to avoid missing
    // optimizations/folds). This is particularly beneficial for scalable
    // vectors that use dynamic tensor shapes.
    auto targetAttr =
        iree_compiler::IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    auto vscaleRange = iree_compiler::getDefaultVscaleRange(targetAttr);
    vector::eliminateVectorMasks(rewriter, funcOp, vscaleRange);
  }

  {
    // Canonicalize mask related ops before we lower them.
    RewritePatternSet maskCanonPatterns(funcOp.getContext());
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        maskCanonPatterns);
    tensor::DimOp::getCanonicalizationPatterns(maskCanonPatterns, context);
    vector::CreateMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                      funcOp.getContext());
    vector::ConstantMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                        funcOp.getContext());
    vector::MaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                funcOp.getContext());
    if (failed(applyPatternsGreedily(funcOp, std::move(maskCanonPatterns)))) {
      return signalPassFailure();
    }
  }

  // TODO: Move this down the pipeline once we have the ODM-based masking
  // representation.
  RewritePatternSet vectorizationPatterns(funcOp.getContext());
  if (generateContract) {
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorizationPatterns);
    vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
    vector::populateSinkVectorOpsPatterns(vectorizationPatterns);
  }
  if (foldCastIntoContract) {
    vector::populateFoldArithExtensionPatterns(vectorizationPatterns);
  }
  if (enableVectorMasking) {
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
        vectorizationPatterns);
    IREE::VectorExt::populateVectorMaskLoweringPatterns(vectorizationPatterns);
    vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                              linalg::LinalgCopyVTWForwardingPattern>(
        funcOp.getContext(), /*benefit=*/2);
  }

  if (enableCleanup) {
    vector::TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                        funcOp.getContext());
    vector::TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                         funcOp.getContext());
  }
  (void)applyPatternsGreedily(funcOp, std::move(vectorizationPatterns));
}

} // namespace
} // namespace mlir::iree_compiler
