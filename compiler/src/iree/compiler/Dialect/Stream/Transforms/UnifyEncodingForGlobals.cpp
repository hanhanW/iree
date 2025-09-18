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
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::Stream {

#define DEBUG_TYPE "iree-stream-unify-encoding-for-globals"

#define GEN_PASS_DEF_UNIFYENCODINGFORGLOBALSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

struct EncodedState : DFX::AbstractState {
  enum State { kIdentity, kEncoded, kMixed, kUnknown };
  State state;

  bool isValidState() const override { return true; }
  bool isAtFixpoint() const override { return state != kUnknown; }

  ChangeStatus indicateOptimisticFixpoint() override {
    return state != kUnknown ? ChangeStatus::UNCHANGED : ChangeStatus::CHANGED;
  }
  ChangeStatus indicatePessimisticFixpoint() override {
    return state == kUnknown ? ChangeStatus::UNCHANGED : ChangeStatus::CHANGED;
  }

#if 0
  ChangeStatus join(const EncodedState &other) {
    if (state == other.state) {
      return ChangeStatus::UNCHANGED;
    }
    if (state == kUnknown || other.state == kUnknown) {
      return ChangeStatus::CHANGED;
    }
    return ChangeStatus::UNCHANGED;
  }
#endif

};

static const std::string getLayoutAsStr(
    const DFX::PotentialValuesState<IREE::Encoding::LayoutAttr> &state,
    AsmState &asmState) {
  std::string str;
  llvm::raw_string_ostream sstream(str);
  sstream << "pvs: ";
  if (state.isValidState()) {
    sstream << "[";
    if (state.isUndefContained()) {
      sstream << "undef, ";
    }
    llvm::interleaveComma(state.getAssumedSet(), sstream,
                          [&](Attribute value) { value.print(sstream); });
    sstream << "]";
  } else {
    sstream << "(invalid)";
  }
  sstream.flush();
  return str;
}

class IdentityLayoutPVS
    : public DFX::StateWrapper<
          DFX::PotentialValuesState<IREE::Encoding::LayoutAttr>,
          DFX::TypedOperationElement<IREE::Util::GlobalOpInterface>> {
public:
  using BaseType = DFX::StateWrapper<
      DFX::PotentialValuesState<IREE::Encoding::LayoutAttr>,
      DFX::TypedOperationElement<IREE::Util::GlobalOpInterface>>;
  using BaseType::BaseType;

  static IdentityLayoutPVS &createForPosition(const Position &pos,
                                              DFX::Solver &solver) {
    return *(new (solver.getAllocator()) IdentityLayoutPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "IdentityLayoutPVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getLayoutAsStr(getState(), asmState);
  }

private:
  void initializeOperation(IREE::Util::GlobalOpInterface globalOp,
                           DFX::Solver &solver) override;
  ChangeStatus updateOperation(IREE::Util::GlobalOpInterface globalOp,
                               DFX::Solver &solver) override;
};
const char IdentityLayoutPVS::ID = 0;

class ValueConsumerLayoutPVS
    : public DFX::StateWrapper<
          DFX::PotentialValuesState<IREE::Encoding::LayoutAttr>,
          DFX::ValueElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<IREE::Encoding::LayoutAttr>,
                        DFX::ValueElement>;
  using BaseType::BaseType;

  static ValueConsumerLayoutPVS &createForPosition(const Position &pos,
                                                   DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueConsumerLayoutPVS(pos));
  }

  // ValueConsumer definitions.
  const std::string getName() const override {
    return "ValueConsumerLayoutPVS";
  }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getLayoutAsStr(getState(), asmState);
  }

private:
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
  TraversalResult updateFromUse(Value value, OpOperand &operand,
                                StateType &newState, DFX::Solver &solver);
};
const char ValueConsumerLayoutPVS::ID = 0;

void ValueConsumerLayoutPVS::initializeValue(Value value, DFX::Solver &solver) {
}

ChangeStatus ValueConsumerLayoutPVS::updateValue(Value value,
                                                 DFX::Solver &solver) {
  StateType newState;
  auto traversalResult = TraversalResult::COMPLETE;

  // Walk into all consumers of the SSA value.
  // Note that we may end up at multiple global stores of different globals
  // by walking down through calls/branches/etc.
  traversalResult |= solver.getExplorer().walkTransitiveUses(
      value,
      [&](OpOperand &operand) {
        traversalResult |= updateFromUse(value, operand, newState, solver);
        return WalkResult::advance();
      },
      (TraversalBehavior::DEFAULT | TraversalBehavior::DONT_WALK_TIED_VALUES));

  if (traversalResult == TraversalResult::INCOMPLETE) {
    // Incomplete traversal because of external call graph edges or pointers.
    newState.unionAssumedWithUndef();
    newState.indicatePessimisticFixpoint();
  }
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

TraversalResult ValueConsumerLayoutPVS::updateFromUse(Value value,
                                                      OpOperand &operand,
                                                      StateType &newState,
                                                      DFX::Solver &solver) {
#if 0
  // If the value is consumed by an affinity-aware op then we can directly use
  // the affinity specified on the op. A majority of the values we care about at
  // the stream level are consumed by affinity-aware ops and earlier in the
  // pipeline dialects may have transfer ops that define affinities we can
  // anchor on.
  if (auto affinityOp =
          dyn_cast<IREE::Stream::AffinityOpInterface>(operand.getOwner())) {
    auto opPVS = solver.getElementFor<OpAffinityPVS>(
        *this, Position::forOperation(operand.getOwner()),
        DFX::Resolution::REQUIRED);
    LLVM_DEBUG({
      llvm::dbgs() << "[ValueConsumerLayoutPVS] value ";
      value.printAsOperand(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << " layout using consumer layout from ";
      opPVS.print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });
    newState ^= opPVS;
  }

  // If the consumer op has the operand tied to one or more results then we walk
  // through to track the transitive consumers. When this analysis runs we are
  // usually still prior to baking out copy-on-write behavior so it's possible
  // that the results of the tied operation end up in different places.
  if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(operand.getOwner())) {
    auto tiedResults = tiedOp.getOperandTiedResults(operand.getOperandNumber());
    for (auto tiedResult : tiedResults) {
      auto resultPVS = solver.getElementFor<ValueConsumerLayoutPVS>(
          *this, Position::forValue(tiedResult), DFX::Resolution::REQUIRED);
      LLVM_DEBUG({
        llvm::dbgs() << "[ValueConsumerLayoutPVS] value ";
        value.printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << " affinity referencing tied operand ";
        operand.get().printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << " result ";
        tiedResult.printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << " as ";
        resultPVS.print(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      newState ^= resultPVS;
    }
  }

  // Handle consumers that are not affinity aware - this should have any control
  // flow ops so that we can track values that flow through the program.
  return TypeSwitch<Operation *, TraversalResult>(operand.getOwner())
      .Case([&](IREE::Stream::AsyncTransferOp op) {
        if (auto targetAffinityAttr = op.getResultAffinityAttr()) {
          LLVM_DEBUG({
            llvm::dbgs() << "[ValueConsumerLayoutPVS] value ";
            value.printAsOperand(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << " affinity unioning with transfer target "
                         << "affinity as " << targetAffinityAttr << "\n";
          });
          newState.unionAssumed(targetAffinityAttr);
        }
        return TraversalResult::COMPLETE;
      })
      .Default([&](Operation *op) { return TraversalResult::COMPLETE; });
#endif
  return TraversalResult::COMPLETE;
}

void IdentityLayoutPVS::initializeOperation(
    IREE::Util::GlobalOpInterface globalOp, DFX::Solver &solver) {
  MLIRContext *ctx = globalOp.getContext();
  Builder b(ctx);
  auto layoutAttr = IREE::Encoding::LayoutAttr::get(
      ctx, b.getArrayAttr({IREE::Encoding::IdentityAttr::get(ctx)}));
  unionAssumed(layoutAttr);
  LLVM_DEBUG({
    llvm::dbgs() << "[IdentityLayoutPVS] global @"
                 << globalOp.getGlobalName().getValue()
                 << " layout explicitly specified as " << layoutAttr;
    llvm::dbgs() << "\n";
  });
  indicateOptimisticFixpoint();
  return;
}

ChangeStatus
IdentityLayoutPVS::updateOperation(IREE::Util::GlobalOpInterface globalOp,
                                   DFX::Solver &solver) {
  StateType newState;
  auto traversalResult = TraversalResult::COMPLETE;

  const auto *globalInfo = solver.getExplorer().getGlobalInfo(globalOp);
  if (globalInfo->isIndirect) {
    traversalResult = TraversalResult::INCOMPLETE;
  }

  for (auto loadOp : globalInfo->getLoads()) {
    auto &valuePVS = solver.getElementFor<ValueConsumerLayoutPVS>(
        *this, Position::forValue(loadOp.getLoadedGlobalValue()),
        DFX::Resolution::OPTIONAL);
    if (valuePVS.isValidState()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[IdentityLayoutPVS] global @"
                     << globalOp.getGlobalName().getValue()
                     << " layout using consumer layout from ";
        valuePVS.print(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      newState ^= valuePVS;
    }
  }

  if (traversalResult == TraversalResult::INCOMPLETE) {
    // Incomplete traversal because of external call graph edges or pointers.
    newState.unionAssumedWithUndef();
    newState.indicatePessimisticFixpoint();
  }
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

class EncodingAnalysis {
public:
  explicit EncodingAnalysis(Operation *rootOp);
  ~EncodingAnalysis() = default;

  // Runs analysis and populates the resource usage map.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run();

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
};

EncodingAnalysis::EncodingAnalysis(Operation *rootOp)
    : explorer(rootOp, TraversalAction::RECURSE), solver(explorer, allocator) {
  explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
      TraversalAction::RECURSE);
  explorer.setDialectAction<mlir::scf::SCFDialect>(TraversalAction::RECURSE);
  explorer.setDialectAction<IREE::Stream::StreamDialect>(
      TraversalAction::RECURSE);
  explorer.setOpAction<IREE::Stream::ExecutableOp>(TraversalAction::IGNORE);
  explorer.initialize();
}

LogicalResult EncodingAnalysis::run() {
  explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
    LDBG() << "iterate " << globalInfo->op;
    if (globalInfo->isIndirect || globalInfo->op.isGlobalMutable()) {
      return;
    }
    LDBG() << "  type: " << globalInfo->op.getGlobalType();
    auto resourceType =
        dyn_cast<IREE::Stream::ResourceType>(globalInfo->op.getGlobalType());
    if (!resourceType) {
      LDBG() << "not resourceType";
      return;
    }
    if (resourceType.getLifetime() != IREE::Stream::Lifetime::Constant) {
      LDBG() << "lifetime is not constant";
      return;
    }
    if (!llvm::hasSingleElement(globalInfo->getStores())) {
      LDBG() << "multiple store";
      return;
    }
    for (auto store : globalInfo->getStores()) {
      if (!store.getStoredGlobalValue()
               .getDefiningOp<IREE::Stream::TensorConstantOp>()) {
        LDBG() << "not start from stream.tensor.constant";
        return;
      }
    }
    LDBG() << "start from constant trait";
    solver.getOrCreateElementFor<IdentityLayoutPVS>(
        Position::forOperation(globalInfo->op));
  });
  return success();
}
}; // namespace

namespace {
struct UnifyEncodingForGlobalsPass
    : public impl::UnifyEncodingForGlobalsPassBase<UnifyEncodingForGlobalsPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    EncodingAnalysis analysis(moduleOp);
    if (failed(analysis.run())) {
      return;
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
