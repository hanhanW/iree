// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-deduplicate-dispatch-result-bindings"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_DEDUPLICATEDISPATCHRESULTBINDINGSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

// Information about a result binding within an executable.
struct ResultBindingInfo {
  // The binding argument index in the function.
  unsigned bindingArgIndex = 0;
  // The result index at dispatch sites (accounting for operand bindings).
  unsigned resultIndex = 0;
  // The tensor value being stored to this binding (from the last store).
  Value storedValue;
};

// Groups of result bindings that store the same tensor value.
struct DuplicateGroup {
  // The binding to keep (first one encountered).
  unsigned keepBindingArgIndex = 0;
  unsigned keepResultIndex = 0;
  // Bindings to remove.
  SmallVector<unsigned> removeBindingArgIndices;
  SmallVector<unsigned> removeResultIndices;
};

// Returns true if the operation is directly in the function's entry block
// (not nested inside any region like scf.for, scf.if, etc.).
// Analyzes an executable function to find result bindings that store the same
// tensor value.
//
// Safety constraints:
// - Exactly one store per binding, in the entry block (not nested in regions)
// - Each store must be a full write (not a partial/tiled write)
// - Only writeonly bindings are considered (not readwrite)
static SmallVector<DuplicateGroup>
analyzeExecutableForDuplicates(mlir::FunctionOpInterface funcOp) {
  Block &entryBlock = funcOp.getFunctionBody().front();

  // Map from binding arg to result binding info.
  DenseMap<BlockArgument, ResultBindingInfo> bindingInfoMap;

  // Find all store operations and their target bindings.
  unsigned resultIndex = 0;
  for (BlockArgument arg : funcOp.getArguments()) {
    if (!isa<BindingType>(arg.getType())) {
      continue;
    }

    // Collect all store operations for this binding.
    SmallVector<TensorExt::DispatchTensorStoreOp> storeOps;
    for (Operation *user : arg.getUsers()) {
      auto subspanOp = dyn_cast<BindingSubspanOp>(user);
      if (!subspanOp) {
        continue;
      }

      for (Operation *subspanUser : subspanOp->getUsers()) {
        auto storeOp = dyn_cast<TensorExt::DispatchTensorStoreOp>(subspanUser);
        if (storeOp && storeOp.getTarget() == subspanOp.getResult()) {
          storeOps.push_back(storeOp);
        }
      }
    }

    // Safety: Skip if not exactly one store.
    if (storeOps.size() != 1) {
      LLVM_DEBUG({
        llvm::dbgs() << "Skipping binding arg " << arg.getArgNumber()
                     << ": has " << storeOps.size() << " stores (expected 1)\n";
      });
      if (!storeOps.empty()) {
        resultIndex++;
      }
      continue;
    }

    TensorExt::DispatchTensorStoreOp storeOp = storeOps[0];

    // Safety: Skip if store is nested inside a region (scf.for, scf.if, etc.).
    // Such stores might not execute due to control flow.
    if (storeOp->getBlock() != &entryBlock) {
      LLVM_DEBUG({
        llvm::dbgs() << "Skipping binding arg " << arg.getArgNumber()
                     << ": store is not in entry block\n";
      });
      resultIndex++;
      continue;
    }

    // Safety: Skip if not a full write (partial/tiled writes not supported).
    if (!storeOp.isStoreToWholeTarget()) {
      LLVM_DEBUG({
        llvm::dbgs() << "Skipping binding arg " << arg.getArgNumber()
                     << ": store is not a full write\n";
      });
      resultIndex++;
      continue;
    }

    // Safety: Only deduplicate writeonly bindings.
    // readwrite bindings are in-out buffers that can't be merged.
    TensorExt::DispatchTensorType targetType = storeOp.getTargetType();
    if (targetType.getAccess() != TensorExt::TensorAccess::WriteOnly) {
      LLVM_DEBUG({
        llvm::dbgs() << "Skipping binding arg " << arg.getArgNumber()
                     << ": not a writeonly binding\n";
      });
      resultIndex++;
      continue;
    }

    ResultBindingInfo info;
    info.bindingArgIndex = arg.getArgNumber();
    info.resultIndex = resultIndex;
    info.storedValue = storeOp.getValue();
    bindingInfoMap[arg] = info;
    resultIndex++;
  }

  // Group bindings by stored value.
  DenseMap<Value, SmallVector<ResultBindingInfo>> valueToBindings;
  for (auto &[arg, info] : bindingInfoMap) {
    valueToBindings[info.storedValue].push_back(info);
  }

  // Build duplicate groups for values stored to multiple bindings.
  SmallVector<DuplicateGroup> duplicateGroups;
  for (auto &[value, bindings] : valueToBindings) {
    if (bindings.size() <= 1) {
      continue;
    }

    // Sort by binding arg index to ensure deterministic behavior.
    // DenseMap iteration order is not guaranteed.
    llvm::sort(bindings,
               [](const ResultBindingInfo &a, const ResultBindingInfo &b) {
                 return a.bindingArgIndex < b.bindingArgIndex;
               });

    DuplicateGroup group;
    // Keep the binding with the lowest arg index.
    group.keepBindingArgIndex = bindings[0].bindingArgIndex;
    group.keepResultIndex = bindings[0].resultIndex;

    // Mark the rest for removal.
    for (size_t i = 1; i < bindings.size(); ++i) {
      group.removeBindingArgIndices.push_back(bindings[i].bindingArgIndex);
      group.removeResultIndices.push_back(bindings[i].resultIndex);
    }

    duplicateGroups.push_back(group);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Found " << duplicateGroups.size()
                 << " duplicate groups in " << funcOp.getName() << "\n";
    for (auto &group : duplicateGroups) {
      llvm::dbgs() << "  Keep binding arg " << group.keepBindingArgIndex
                   << " (result " << group.keepResultIndex << "), remove: ";
      llvm::interleaveComma(group.removeBindingArgIndices, llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  return duplicateGroups;
}

// Verifies that deduplication is safe across all dispatch sites.
// Returns true if all dispatch sites have uniform properties for the duplicate
// group (same device affinity, encodings, etc.).
static bool verifyDispatchSiteUniformity(
    ArrayRef<TensorDispatchOp> dispatchOps,
    const SmallVector<DuplicateGroup> &duplicateGroups) {
  if (dispatchOps.empty() || duplicateGroups.empty()) {
    return true;
  }

  for (const auto &group : duplicateGroups) {
    for (auto dispatchOp : dispatchOps) {
      // Check that the result encodings match between kept and removed results.
      ArrayAttr resultEncodings = dispatchOp.getResultEncodings();
      if (group.keepResultIndex >= resultEncodings.size()) {
        return false;
      }
      Attribute keepEncoding = resultEncodings[group.keepResultIndex];

      for (unsigned removeIdx : group.removeResultIndices) {
        if (removeIdx >= resultEncodings.size()) {
          return false;
        }
        Attribute removeEncoding = resultEncodings[removeIdx];
        if (keepEncoding != removeEncoding) {
          LLVM_DEBUG(llvm::dbgs() << "Encoding mismatch: keep=" << keepEncoding
                                  << " remove=" << removeEncoding << "\n");
          return false;
        }
      }

      // Check that result sizes match.
      Value keepSize = dispatchOp.getResultSize(group.keepResultIndex);
      for (unsigned removeIdx : group.removeResultIndices) {
        Value removeSize = dispatchOp.getResultSize(removeIdx);
        if (keepSize != removeSize) {
          LLVM_DEBUG(llvm::dbgs() << "Size mismatch for results\n");
          return false;
        }
      }
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Transformation
//===----------------------------------------------------------------------===//

// Updates the executable function to remove duplicate result bindings.
static void
updateExecutableFunction(mlir::FunctionOpInterface funcOp,
                         const SmallVector<DuplicateGroup> &duplicateGroups) {
  // Collect all binding args to remove.
  llvm::DenseSet<unsigned> argsToRemove;
  for (const auto &group : duplicateGroups) {
    for (unsigned argIdx : group.removeBindingArgIndices) {
      argsToRemove.insert(argIdx);
    }
  }

  Block &entryBlock = funcOp.getFunctionBody().front();

  // Remove all operations that use the duplicate bindings.
  // A binding may have multiple stores (e.g., intermediate stores before the
  // final one), so we need to remove all of them, not just the last store.
  for (BlockArgument arg : entryBlock.getArguments()) {
    if (!argsToRemove.contains(arg.getArgNumber())) {
      continue;
    }
    // For each subspan of this binding, erase all its users (stores), then
    // erase the subspan itself.
    for (Operation *user : llvm::make_early_inc_range(arg.getUsers())) {
      // Erase all users of the subspan (should be store/load ops).
      for (Operation *subspanUser :
           llvm::make_early_inc_range(user->getUsers())) {
        subspanUser->erase();
      }
      // Erase the subspan.
      user->erase();
    }
  }

  // Erase the duplicate binding arguments.
  entryBlock.eraseArguments([&](BlockArgument arg) {
    return argsToRemove.contains(arg.getArgNumber());
  });

  // Update function type.
  funcOp.setType(FunctionType::get(funcOp.getContext(),
                                   entryBlock.getArgumentTypes(), {}));
}

// Updates a dispatch site to remove duplicate results.
static void
updateDispatchSite(TensorDispatchOp dispatchOp,
                   const SmallVector<DuplicateGroup> &duplicateGroups) {
  if (duplicateGroups.empty()) {
    return;
  }

  // Build mapping from old result index to new result index.
  unsigned numOldResults = dispatchOp.getNumResults();
  llvm::DenseSet<unsigned> resultsToRemove;
  DenseMap<unsigned, unsigned> oldToKeptResult;

  for (const auto &group : duplicateGroups) {
    for (unsigned removeIdx : group.removeResultIndices) {
      resultsToRemove.insert(removeIdx);
      oldToKeptResult[removeIdx] = group.keepResultIndex;
    }
  }

  // Replace uses of removed results with kept results.
  for (const auto &group : duplicateGroups) {
    Value keptResult = dispatchOp.getResult(group.keepResultIndex);
    for (unsigned removeIdx : group.removeResultIndices) {
      Value removedResult = dispatchOp.getResult(removeIdx);
      removedResult.replaceAllUsesWith(keptResult);
    }
  }

  // Build new operand/result lists excluding removed results.
  SmallVector<Value> newResultSizes;
  SmallVector<Attribute> newResultEncodings;
  SmallVector<Value> newResultEncodingDims;
  SmallVector<Type> newResultTypes;

  // Track which old result indices map to which new result indices.
  DenseMap<unsigned, unsigned> oldToNewResultIndex;
  unsigned newResultIndex = 0;

  ArrayRef<Attribute> oldResultEncodings =
      dispatchOp.getResultEncodings().getValue();
  OperandRange oldResultSizes = dispatchOp.getResultSizes();
  OperandRange oldResultEncodingDims = dispatchOp.getResultEncodingDims();

  // Track the current position in the flat result_encoding_dims list.
  unsigned dimOffset = 0;

  for (unsigned i = 0; i < numOldResults; ++i) {
    // Get the tensor type from the encoding to count dynamic dims.
    auto encodingType = cast<RankedTensorType>(
        cast<TypeAttr>(oldResultEncodings[i]).getValue());
    unsigned numDynamicDims = encodingType.getNumDynamicDims();

    if (resultsToRemove.contains(i)) {
      // Skip this result and its dims.
      dimOffset += numDynamicDims;
      continue;
    }

    oldToNewResultIndex[i] = newResultIndex++;
    newResultTypes.push_back(dispatchOp.getResult(i).getType());
    newResultEncodings.push_back(oldResultEncodings[i]);
    newResultSizes.push_back(oldResultSizes[i]);

    // Copy the dynamic dims for this result.
    for (unsigned d = 0; d < numDynamicDims; ++d) {
      newResultEncodingDims.push_back(oldResultEncodingDims[dimOffset + d]);
    }
    dimOffset += numDynamicDims;
  }

  // Handle tied operands - need to remap indices.
  // For now, we skip deduplication if any result is tied to an operand.
  // TODO: Handle tied operands properly by remapping indices.
  ArrayAttr newTiedOperandsAttr;
  if (std::optional<ArrayAttr> oldTiedOperands = dispatchOp.getTiedOperands()) {
    SmallVector<int64_t> newTiedOperands;
    for (auto [oldIdx, tiedAttr] :
         llvm::enumerate(oldTiedOperands->getValue())) {
      if (resultsToRemove.contains(oldIdx)) {
        continue;
      }
      newTiedOperands.push_back(cast<IntegerAttr>(tiedAttr).getInt());
    }
    OpBuilder attrBuilder(dispatchOp.getContext());
    // Use getIndexArrayAttr because Util_TiedOpStorageAttr expects IndexType.
    newTiedOperandsAttr = attrBuilder.getIndexArrayAttr(newTiedOperands);
  }

  // Create new dispatch op.
  OpBuilder builder(dispatchOp);
  auto newDispatchOp = TensorDispatchOp::create(
      builder, dispatchOp.getLoc(), newResultTypes, dispatchOp.getWorkload(),
      dispatchOp.getEntryPointsAttr(), dispatchOp.getMixedOperands(),
      dispatchOp.getOperandSizes(), dispatchOp.getOperandEncodingsAttr(),
      dispatchOp.getOperandEncodingDims(), newResultSizes,
      builder.getArrayAttr(newResultEncodings), newResultEncodingDims,
      newTiedOperandsAttr, dispatchOp.getAffinityAttr());

  // Replace uses of old results with new results.
  for (auto [oldIdx, newIdx] : oldToNewResultIndex) {
    dispatchOp.getResult(oldIdx).replaceAllUsesWith(
        newDispatchOp.getResult(newIdx));
  }

  dispatchOp.erase();
}

// Deduplicates result bindings for a single executable export.
static void deduplicateResultBindings(ExecutableOp executableOp,
                                      ExecutableExportOp exportOp,
                                      ArrayRef<TensorDispatchOp> dispatchOps) {
  if (dispatchOps.empty()) {
    return;
  }

  // Get the function for this export.
  mlir::FunctionOpInterface funcOp = exportOp.lookupFunctionRef();
  if (!funcOp) {
    return;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "---- deduplicateResultBindings(@"
                 << executableOp.getSymName() << "::" << exportOp.getSymName()
                 << ") ----\n";
  });

  // Analyze the executable for duplicate result bindings.
  SmallVector<DuplicateGroup> duplicateGroups =
      analyzeExecutableForDuplicates(funcOp);
  if (duplicateGroups.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No duplicates found\n");
    return;
  }

  // Verify that deduplication is safe across all dispatch sites.
  if (!verifyDispatchSiteUniformity(dispatchOps, duplicateGroups)) {
    LLVM_DEBUG(llvm::dbgs() << "  Dispatch sites not uniform, skipping\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "  Deduplicating " << duplicateGroups.size()
                          << " groups\n");

  // Update the executable function.
  updateExecutableFunction(funcOp, duplicateGroups);

  // Update all dispatch sites.
  for (auto dispatchOp : dispatchOps) {
    updateDispatchSite(dispatchOp, duplicateGroups);
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct DeduplicateDispatchResultBindingsPass
    : public impl::DeduplicateDispatchResultBindingsPassBase<
          DeduplicateDispatchResultBindingsPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // Gather all tensor dispatch ops and bucket by entry point.
    DenseMap<Operation *, SmallVector<TensorDispatchOp>> entryDispatchMap;
    moduleOp->walk([&](TensorDispatchOp dispatchOp) {
      dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
        Operation *exportOp =
            symbolTable.lookupNearestSymbolFrom(dispatchOp, entryPointAttr);
        entryDispatchMap[exportOp].push_back(dispatchOp);
      });
    });

    // Process each executable.
    for (auto executableOp : moduleOp.getBodyRegion().getOps<ExecutableOp>()) {
      if (!executableOp.getInnerModule()) {
        continue;
      }
      for (auto exportOp : executableOp.getOps<ExecutableExportOp>()) {
        deduplicateResultBindings(executableOp, exportOp,
                                  entryDispatchMap[exportOp]);
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
