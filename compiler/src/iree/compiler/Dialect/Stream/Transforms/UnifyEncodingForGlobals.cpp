// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::Stream {

#define DEBUG_TYPE "iree-stream-specialize-encodings"

#define GEN_PASS_DEF_UNIFYENCODINGFORGLOBALSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {
class EncodingAnalysis {
public:
  explicit EncodingAnalysis(ModuleOp moduleOp);
  ~EncodingAnalysis();

  // Runs analysis and populates the resource usage map.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run();

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
};
}; // namespace

namespace {
struct UnifyEncodingForGlobalsPass
    : public impl::UnifyEncodingForGlobalsPassBase<UnifyEncodingForGlobalsPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    (void)moduleOp;
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
