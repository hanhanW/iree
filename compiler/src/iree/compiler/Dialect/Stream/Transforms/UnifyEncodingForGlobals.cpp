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
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

namespace mlir::iree_compiler::IREE::Stream {

#define DEBUG_TYPE "iree-stream-unify-encoding-for-globals"

#define GEN_PASS_DEF_UNIFYENCODINGFORGLOBALSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

// TODO: Revisit the comments. Some of them are either copy from existing code
// or self note.

// [Global, direct encoding layout]
// TODO: Tracking the encoding chain may not be needed at this level. It makes
// sense to use the last encoding instead, since encoding is a hint. You don't
// expect to relayout a tensor several times in the encode ops chain.
struct ParameterWrapper {
  explicit ParameterWrapper(Operation *rootOp) {
    if (auto global =
            dyn_cast_if_present<IREE::Util::GlobalOpInterface>(rootOp)) {
      op = global;
    } else {
      // assert(false);
    }
  }
  explicit ParameterWrapper(IREE::Util::GlobalOpInterface rootOp)
      : op(rootOp) {}
  explicit ParameterWrapper(std::string str) : parameter(str) {}
  explicit ParameterWrapper(StringRef str) : parameter(str) {}

  IREE::Util::GlobalOpInterface op;
  std::string parameter;
};
bool operator==(const ParameterWrapper &lhs, const ParameterWrapper &rhs) {
  return std::tie(lhs.op, lhs.parameter) == std::tie(rhs.op, rhs.parameter);
}
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ParameterWrapper &wrapper) {
  if (wrapper.op) {
    os << "global(" << wrapper.op << ")";
  } else {
    os << "parameter(" << wrapper.parameter << ")";
  }
  return os;
}
} // namespace
} // namespace mlir::iree_compiler::IREE::Stream
namespace llvm {
using mlir::iree_compiler::IREE::Stream::ParameterWrapper;
template <>
struct DenseMapInfo<ParameterWrapper> {
  static inline ParameterWrapper getEmptyKey() {
    ParameterWrapper empty(nullptr);
    empty.parameter = "";
    return empty;
  }
  static inline ParameterWrapper getTombstoneKey() {
    ParameterWrapper empty(nullptr);
    empty.parameter = "";
    return empty;
  }
  static unsigned getHashValue(const ParameterWrapper &pos) {
    if (pos.op) {
      return DenseMapInfo<void *>::getHashValue(pos.op);
    } else {
      return DenseMapInfo<StringRef>::getHashValue(pos.parameter);
    }
    assert(false);
    return 0;
  }

  static bool isEqual(const ParameterWrapper &a, const ParameterWrapper &b) {
    return a == b;
  }
};
} // namespace llvm

namespace mlir::iree_compiler::IREE::Stream {
namespace {
using Item = std::tuple<ParameterWrapper, SmallVector<Attribute>>;

static const std::string
getLayoutSetAsStr(const DFX::PotentialValuesState<Item> &state,
                  AsmState &asmState) {
  DenseSet<Item, DenseMapInfo<Item>> assumedSet = state.getAssumedSet();
  std::string str;
  llvm::raw_string_ostream sstream(str);
  sstream << "pvs: ";
  if (state.isValidState()) {
    sstream << "[";
    if (state.isUndefContained()) {
      sstream << "undef, ";
    }
    llvm::interleaveComma(state.getAssumedSet(), sstream, [&](Item item) {
      sstream << "\n\t" << std::get<0>(item);
      sstream << ", chain=(";
      llvm::interleaveComma(std::get<1>(item), sstream,
                            [&](Attribute value) { value.print(sstream); });
      sstream << ")";
    });
    sstream << "]";
  } else {
    sstream << "(invalid)";
  }
  sstream.flush();
  return str;
}

class GlobalPVS
    : public DFX::StateWrapper<
          DFX::PotentialValuesState<Item>,
          DFX::TypedOperationElement<IREE::Util::GlobalOpInterface>> {
public:
  using BaseType = DFX::StateWrapper<
      DFX::PotentialValuesState<Item>,
      DFX::TypedOperationElement<IREE::Util::GlobalOpInterface>>;
  using BaseType::BaseType;

  static GlobalPVS &createForPosition(const Position &pos,
                                      DFX::Solver &solver) {
    return *(new (solver.getAllocator()) GlobalPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "GlobalPVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getLayoutSetAsStr(getState(), asmState);
  }

private:
  void initializeOperation(IREE::Util::GlobalOpInterface globalOp,
                           DFX::Solver &solver) override;
  ChangeStatus updateOperation(IREE::Util::GlobalOpInterface globalOp,
                               DFX::Solver &solver) override;
};
const char GlobalPVS::ID = 0;

class OpPVS : public DFX::StateWrapper<DFX::PotentialValuesState<Item>,
                                       DFX::OperationElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<Item>, DFX::OperationElement>;
  using BaseType::BaseType;

  static OpPVS &createForPosition(const Position &pos, DFX::Solver &solver) {
    return *(new (solver.getAllocator()) OpPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "OpPVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getLayoutSetAsStr(getState(), asmState);
  }

private:
  void initializeOperation(Operation *op, DFX::Solver &solver) override;
  ChangeStatus updateOperation(Operation *op, DFX::Solver &solver) override;
};
const char OpPVS::ID = 0;

class DirectOpPVS : public DFX::StateWrapper<DFX::PotentialValuesState<Item>,
                                       DFX::OperationElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<Item>, DFX::OperationElement>;
  using BaseType::BaseType;

  static DirectOpPVS &createForPosition(const Position &pos, DFX::Solver &solver) {
    return *(new (solver.getAllocator()) DirectOpPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "DirectOpPVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getLayoutSetAsStr(getState(), asmState);
  }

private:
  void initializeOperation(Operation *op, DFX::Solver &solver) override;
  ChangeStatus updateOperation(Operation *op, DFX::Solver &solver) override;
};
const char DirectOpPVS::ID = 0;

class ValueProducerPVS
    : public DFX::StateWrapper<DFX::PotentialValuesState<Item>,
                               DFX::ValueElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<Item>, DFX::ValueElement>;
  using BaseType::BaseType;

  static ValueProducerPVS &createForPosition(const Position &pos,
                                             DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueProducerPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "ValueProducerPVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getLayoutSetAsStr(getState(), asmState);
  }

private:
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
};
const char ValueProducerPVS::ID = 0;

class DirectValueProducerPVS
    : public DFX::StateWrapper<DFX::PotentialValuesState<Item>,
                               DFX::ValueElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<Item>, DFX::ValueElement>;
  using BaseType::BaseType;

  static DirectValueProducerPVS &createForPosition(const Position &pos,
                                                   DFX::Solver &solver) {
    return *(new (solver.getAllocator()) DirectValueProducerPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override {
    return "DirectValueProducerPVS";
  }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getLayoutSetAsStr(getState(), asmState);
  }

private:
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
};
const char DirectValueProducerPVS::ID = 0;

#if 0
class ValueConsumerPVS
    : public DFX::StateWrapper<DFX::PotentialValuesState<Item>,
                               DFX::ValueElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<Item>, DFX::ValueElement>;
  using BaseType::BaseType;

  static ValueConsumerPVS &createForPosition(const Position &pos,
                                             DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueConsumerPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "ValueConsumerPVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getLayoutSetAsStr(getState(), asmState);
  }

private:
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
};
const char ValueConsumerPVS::ID = 0;
#endif

//===----------------------------------------------------------------------===//
// GlobalPVS
//===----------------------------------------------------------------------===//

void GlobalPVS::initializeOperation(IREE::Util::GlobalOpInterface globalOp,
                                    DFX::Solver &solver) {
  auto *globalInfo = solver.getExplorer().getGlobalInfo(globalOp);
  if (!globalInfo || globalInfo->isIndirect) {
    // Cannot perform analysis.
    indicatePessimisticFixpoint();
  } else if (globalInfo) {
    Item init = {ParameterWrapper(globalInfo->op), {}};
    unionAssumed(init);
  }
  return;
}

ChangeStatus GlobalPVS::updateOperation(IREE::Util::GlobalOpInterface globalOp,
                                        DFX::Solver &solver) {
  StateType newState;
  auto traversalResult = TraversalResult::COMPLETE;

  const auto *globalInfo = solver.getExplorer().getGlobalInfo(globalOp);
  if (globalInfo->isIndirect) {
    traversalResult = TraversalResult::INCOMPLETE;
  }
  for (auto store : globalInfo->getStores()) {
    newState.unionAssumed(
        solver.getOrCreateElementFor<OpPVS>(Position::forOperation(store)));
  }
  if (traversalResult == TraversalResult::INCOMPLETE) {
    // Incomplete traversal because of external call graph edges or pointers.
    newState.unionAssumedWithUndef();
    newState.indicatePessimisticFixpoint();
  }
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

//===----------------------------------------------------------------------===//
// OpPVS
//===----------------------------------------------------------------------===//

void OpPVS::initializeOperation(Operation *op, DFX::Solver &solver) {}

ChangeStatus OpPVS::updateOperation(Operation *op, DFX::Solver &solver) {
  StateType newState;
  TypeSwitch<Operation *>(op)
      .Case<IREE::Util::GlobalStoreOpInterface>([&](auto store) {
        auto &producerPVS = solver.getElementFor<ValueProducerPVS>(
            *this, Position::forValue(store.getStoredGlobalValue()),
            DFX::Resolution::REQUIRED);
        LLVM_DEBUG(producerPVS.getAsStr(solver.getAsmState()));
        newState.unionAssumed(producerPVS.getState());
        if (producerPVS.isValidState()) {
          newState.unionAssumed(producerPVS);
        } else {
          newState.unionAssumedWithUndef();
          newState.indicatePessimisticFixpoint();
        }
      })
      .Case<IREE::Stream::TensorDispatchOp>([&](auto dispatchOp) {
        for (Value operand : dispatchOp.getMixedOperands()) {
          auto &producerPVS = solver.getElementFor<ValueProducerPVS>(
              *this, Position::forValue(operand), DFX::Resolution::OPTIONAL);
          LLVM_DEBUG(producerPVS.getAsStr(solver.getAsmState()));
          newState.unionAssumed(producerPVS.getState());
          if (producerPVS.isValidState()) {
            newState.unionAssumed(producerPVS);
          } else {
            newState.unionAssumedWithUndef();
            newState.indicatePessimisticFixpoint();
          }
        }
      })
      .Case<IREE::Stream::AsyncTransferOp>([&](auto op) {
        auto sourceState = solver.getElementFor<ValueProducerPVS>(
            *this, Position::forValue(op.getSource()),
            DFX::Resolution::OPTIONAL);
        newState ^= sourceState;
      })
      .Default([](Operation *) { return; });
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

//===----------------------------------------------------------------------===//
// DirectOpPVS
//===----------------------------------------------------------------------===//

void DirectOpPVS::initializeOperation(Operation *op, DFX::Solver &solver) {}

ChangeStatus DirectOpPVS::updateOperation(Operation *op, DFX::Solver &solver) {
  StateType newState;
  TypeSwitch<Operation *>(op)
      .Case<IREE::Stream::TensorDispatchOp>([&](auto dispatchOp) {
        for (Value operand : dispatchOp.getMixedOperands()) {
          auto &producerPVS = solver.getElementFor<DirectValueProducerPVS>(
              *this, Position::forValue(operand), DFX::Resolution::OPTIONAL);
          LLVM_DEBUG(producerPVS.getAsStr(solver.getAsmState()));
          newState.unionAssumed(producerPVS.getState());
          if (producerPVS.isValidState()) {
            newState.unionAssumed(producerPVS);
          } else {
            newState.unionAssumedWithUndef();
            newState.indicatePessimisticFixpoint();
          }
        }
      })
      .Case<IREE::Stream::AsyncTransferOp>([&](auto op) {
        auto sourceState = solver.getElementFor<DirectValueProducerPVS>(
            *this, Position::forValue(op.getSource()),
            DFX::Resolution::OPTIONAL);
        newState ^= sourceState;
      })
      .Default([](Operation *) { return; });
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

//===----------------------------------------------------------------------===//
// ValueProducerPVS
//===----------------------------------------------------------------------===//

void ValueProducerPVS::initializeValue(Value value, DFX::Solver &solver) {}

ChangeStatus ValueProducerPVS::updateValue(Value value, DFX::Solver &solver) {
  MLIRContext *ctx = value.getContext();
  StateType newState;
  auto traversalResult = TraversalResult::COMPLETE;
  traversalResult |= solver.getExplorer().walkDefiningOps(
      value,
      [&](OpResult result) {
        if (isa<CallOpInterface>(result.getOwner())) {
          return WalkResult::advance();
        }
        LDBG() << "walk " << value;
        LDBG() << "\tmay be defined by: " << *result.getOwner();

        // TODO: Handle interface ops, if any.

        // Special handling for specific ops.
        TypeSwitch<Operation *>(result.getOwner())
            .Case<IREE::Util::GlobalLoadOpInterface>([&](auto loadOp) {
              const Explorer::GlobalInfo *globalInfo =
                  solver.getExplorer().queryGlobalInfoFrom(
                      loadOp.getGlobalName(), loadOp);
              auto &globalPVS = solver.getElementFor<GlobalPVS>(
                  *this, Position::forOperation(globalInfo->op),
                  DFX::Resolution::REQUIRED);
              newState.unionAssumed(globalPVS);
              //Item item = {ParameterWrapper(globalInfo->op), {}};
              //newState.unionAssumed(item);
            })
            .Case<IREE::Stream::AsyncTransferOp>([&](auto op) {
              auto sourceState = solver.getElementFor<ValueProducerPVS>(
                  *this, Position::forValue(op.getSource()),
                  DFX::Resolution::REQUIRED);
              newState ^= sourceState;
            })
            .Case<IREE::Stream::TensorEncodeOp>([&](auto op) {
              auto sourceState = solver.getElementFor<ValueProducerPVS>(
                  *this, Position::forValue(op.getSource()),
                  DFX::Resolution::REQUIRED);
              auto encodingType =
                  dyn_cast<RankedTensorType>(op.getResultEncoding());
              if (!encodingType) {
                // Bail out if we don't know what to do.
                return;
              }
              Attribute encoding = encodingType.getEncoding();
              if (!encoding) {
                encoding = IREE::Encoding::IdentityAttr::get(ctx);
              }
              for (auto [globalOp, encodingChain] :
                   sourceState.getState().getAssumedSet()) {
                Item item = {globalOp, encodingChain};
                std::get<1>(item).push_back(encoding);
                newState.unionAssumed(item);
              }
            })
            .Case<IREE::Stream::TensorConstantOp>([&](auto op) {
              if (auto attr = dyn_cast<IREE::Stream::NamedParameterAttr>(
                      op.getValue())) {
                Item item = {ParameterWrapper(attr.getKey()), {}};
                newState.unionAssumed(item);
              }
            })
            // XXX: It looks wrong to me, because it can result in different
            // global if the dispatch op is involved.
            .Case<IREE::Stream::TensorDispatchOp>([&](auto op) {
              // TODO: Properly filter the ops.
              if (!isa_and_nonnull<IREE::Util::InitializerOp>(
                      op->getParentOp())) {
                return;
              }
              for (Value operand : op.getMixedOperands()) {
                auto sourceState = solver.getElementFor<ValueProducerPVS>(
                    *this, Position::forValue(operand),
                    DFX::Resolution::REQUIRED);
                newState ^= sourceState;
              }
            })
            .Default([&](auto op) {});

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

//===----------------------------------------------------------------------===//
// DirectValueProducerPVS
//===----------------------------------------------------------------------===//

void DirectValueProducerPVS::initializeValue(Value value, DFX::Solver &solver) {}

ChangeStatus DirectValueProducerPVS::updateValue(Value value,
                                                 DFX::Solver &solver) {
  MLIRContext *ctx = value.getContext();
  StateType newState;
  Operation *producer = value.getDefiningOp();
  if (!producer) {
    return DFX::clampStateAndIndicateChange(getState(), newState);
  }
  TypeSwitch<Operation *>(producer)
      .Case<IREE::Util::GlobalLoadOpInterface>([&](auto loadOp) {
        const Explorer::GlobalInfo *globalInfo =
            solver.getExplorer().queryGlobalInfoFrom(loadOp.getGlobalName(),
                                                     loadOp);
        Item item = {ParameterWrapper(globalInfo->op), {}};
        newState.unionAssumed(item);
      })
      .Case<IREE::Stream::AsyncTransferOp>([&](auto op) {
        auto sourceState = solver.getElementFor<DirectValueProducerPVS>(
            *this, Position::forValue(op.getSource()),
            DFX::Resolution::REQUIRED);
        newState ^= sourceState;
      })
      .Case<IREE::Stream::TensorEncodeOp>([&](auto op) {
        auto sourceState = solver.getElementFor<DirectValueProducerPVS>(
            *this, Position::forValue(op.getSource()),
            DFX::Resolution::REQUIRED);
        auto encodingType = dyn_cast<RankedTensorType>(op.getResultEncoding());
        if (!encodingType) {
          // Bail out if we don't know what to do.
          return;
        }
        Attribute encoding = encodingType.getEncoding();
        if (!encoding) {
          encoding = IREE::Encoding::IdentityAttr::get(ctx);
        }
        for (auto [globalOp, encodingChain] :
             sourceState.getState().getAssumedSet()) {
          Item item = {globalOp, encodingChain};
          std::get<1>(item).push_back(encoding);
          newState.unionAssumed(item);
        }
      })
      .Case<IREE::Stream::TensorConstantOp>([&](auto op) {
        if (auto attr =
                dyn_cast<IREE::Stream::NamedParameterAttr>(op.getValue())) {
          Item item = {ParameterWrapper(attr.getKey()), {}};
          newState.unionAssumed(item);
        }
      })
      .Default([&](auto op) {});

  return DFX::clampStateAndIndicateChange(getState(), newState);
}

//===----------------------------------------------------------------------===//
// ValueConsumerPVS
//===----------------------------------------------------------------------===//

#if 0
void ValueConsumerPVS::initializeValue(Value value, DFX::Solver &solver) {}

ChangeStatus ValueConsumerPVS::updateValue(Value value, DFX::Solver &solver) {
  MLIRContext *ctx = value.getContext();
  StateType newState;
  auto traversalResult = TraversalResult::COMPLETE;
  traversalResult |= solver.getExplorer().walkTransitiveUses(
      value,
      [&](OpOperand &operand) {
        // Special handling for specific ops.
        TypeSwitch<Operation *>(operand.get().getDefiningOp())
            .Case<IREE::Util::GlobalLoadOpInterface>([&](auto loadOp) {
              const Explorer::GlobalInfo *globalInfo =
                  solver.getExplorer().queryGlobalInfoFrom(
                      loadOp.getGlobalName(), loadOp);
              auto &globalPVS = solver.getElementFor<ValueConsumerPVS>(
                  *this, Position::forOperation(globalInfo->op),
                  DFX::Resolution::OPTIONAL);
              newState.unionAssumed(globalPVS);
            })
            .Case<IREE::Stream::AsyncTransferOp>([&](auto op) {
              auto sourceState = solver.getElementFor<ValueConsumerPVS>(
                  *this, Position::forValue(op.getSource()),
                  DFX::Resolution::OPTIONAL);
              newState ^= sourceState;
            })
            .Case<IREE::Stream::TensorEncodeOp>([&](auto op) {
              auto sourceState = solver.getElementFor<ValueConsumerPVS>(
                  *this, Position::forValue(op.getSource()),
                  DFX::Resolution::OPTIONAL);
              auto encodingType =
                  dyn_cast<RankedTensorType>(op.getResultEncoding());
              if (!encodingType) {
                // Bail out if we don't know what to do.
                return;
              }
              Attribute encoding = encodingType.getEncoding();
              if (!encoding) {
                encoding = IREE::Encoding::IdentityAttr::get(ctx);
              }
              for (auto [globalOp, encodingChain] :
                   sourceState.getState().getAssumedSet()) {
                Item item = {globalOp, encodingChain};
                std::get<1>(item).push_back(encoding);
                newState.unionAssumed(item);
              }
            })
            //.Case<IREE::Stream::TensorDispatchOp>([&](auto op) {

            //})
            .Default([&](auto op) {});

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
#endif

//===----------------------------------------------------------------------===//
// EncodingAnalysis
//===----------------------------------------------------------------------===//

class EncodingAnalysis {
public:
  explicit EncodingAnalysis(Operation *rootOp);
  ~EncodingAnalysis() = default;

  LogicalResult run();

  SmallVector<ParameterWrapper> getGlobalOps() {
    return llvm::to_vector(globalConsumerLayouts.keys());
  }

  SmallVector<Item> lookupGlobalConsumerLayouts(ParameterWrapper op);

  SmallVector<ParameterWrapper>
  lookupGlobalInputs(IREE::Stream::TensorDispatchOp dispatchOp);

  std::optional<ParameterWrapper> getSourceGlobal(ParameterWrapper op);

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
  SmallVector<IREE::Util::GlobalOpInterface> globalOps;
  DenseMap<ParameterWrapper, llvm::SmallSetVector<Item, 4>>
      globalConsumerLayouts;
  DenseMap<ParameterWrapper, SmallVector<IREE::Stream::TensorDispatchOp>>
      tensorDispatchUsersFromGlobal;
};
}; // namespace

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
    // TODO: Revisit the check, since it is copied from AffinityAnalysis and the
    // toy IR does not have the case.
    if (globalInfo->isIndirect || globalInfo->op.isGlobalMutable()) {
      return;
    }
    if (!isa<IREE::Stream::ResourceType>(globalInfo->op.getGlobalType())) {
      return;
    }
    solver.getOrCreateElementFor<GlobalPVS>(
        Position::forOperation(globalInfo->op));
    globalOps.push_back(globalInfo->op);
  });

  explorer.forEachFunctionLikeOp([&](FunctionOpInterface funcOp) {
    funcOp.walk([&](IREE::Stream::TensorDispatchOp op) {
      solver.getOrCreateElementFor<DirectOpPVS>(Position::forOperation(op));
    });
  });

  // TODO: Expose it as an CLI option.
  constexpr int64_t kSolverMaxIterations = 10;
  if (failed(solver.run(kSolverMaxIterations))) {
    return failure(); // did not converge
  }

  LLVM_DEBUG({
    llvm::dbgs()
        << "\n\n[Analysis] encoding analysis results for the whole module:\n";
    solver.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  for (IREE::Util::GlobalOpInterface globalOp : globalOps) {
    auto globalPVS =
        solver.lookupElementFor<GlobalPVS>(Position::forOperation(globalOp));
    for (auto [sourceOp, encodings] : globalPVS->getAssumedSet()) {
      globalConsumerLayouts[sourceOp].insert(
          {ParameterWrapper(globalOp), encodings});
      LLVM_DEBUG({
        llvm::dbgs() << "GlobalOp:" << sourceOp << "\n";
        llvm::dbgs() << "\thas consumer: " << globalOp.getGlobalName() << "\n";
        llvm::dbgs() << "\twhere the encoding is (";
        llvm::interleaveComma(encodings, llvm::dbgs(), [&](Attribute value) {
          value.print(llvm::dbgs());
        });
        llvm::dbgs() << ")\n";
      });
    }
  }

  explorer.forEachFunctionLikeOp([&](FunctionOpInterface funcOp) {
    // Skip Initializer for now.
    // TODO: Check if we need it.
    if (isa<IREE::Util::InitializerOp>(funcOp)) {
      return;
    }
    funcOp.walk([&](IREE::Stream::TensorDispatchOp op) {
      SmallVector<ParameterWrapper> globals = lookupGlobalInputs(op);
      LLVM_DEBUG({
        llvm::dbgs() << "TensorDispatchOp: " << op << "\n";
        for (auto global : globals) {
          llvm::dbgs() << "\t" << global << "\n";
        }
      });
      for (auto global : globals) {
        tensorDispatchUsersFromGlobal[global].push_back(op);
      }
    });
  });

  return success();
}

SmallVector<ParameterWrapper> EncodingAnalysis::lookupGlobalInputs(
    IREE::Stream::TensorDispatchOp dispatchOp) {
  llvm::SmallSetVector<ParameterWrapper, 4> result;
  auto dispatchOpPVS = solver.getOrCreateElementFor<DirectOpPVS>(
      Position::forOperation(dispatchOp));
  for (auto [globalOp, encodings] : dispatchOpPVS.getAssumedSet()) {
    result.insert(globalOp);
  }
  return result.takeVector();
}

SmallVector<Item>
EncodingAnalysis::lookupGlobalConsumerLayouts(ParameterWrapper op) {
  return globalConsumerLayouts[op].takeVector();
}

std::optional<ParameterWrapper>
EncodingAnalysis::getSourceGlobal(ParameterWrapper op) {
  if (!op.op) {
    // Parameter is always the source.
    return op;
  }
  std::optional<ParameterWrapper> result;
  auto globalPVS =
      solver.lookupElementFor<GlobalPVS>(Position::forOperation(op.op));
  LLVM_DEBUG(llvm::dbgs() << "Looking up source global for:"
                          << *op.op.getOperation() << "\n";);
  for (auto [sourceOp, encodings] : globalPVS->getAssumedSet()) {
    if (sourceOp == op) {
      LLVM_DEBUG(llvm::dbgs() << "Skip self as source" << "\n");
      continue;
    }
    if (!encodings.empty()) {
      continue;
    }
    LLVM_DEBUG({ llvm::dbgs() << "Candidate:" << sourceOp << "\n"; });
    if (result) {
      assert(false && "expect to be initialized once");
      return std::nullopt;
    }
    result = sourceOp;
  }
  return result;
}

namespace {
struct UnifyEncodingForGlobalsPass
    : public impl::UnifyEncodingForGlobalsPassBase<
          UnifyEncodingForGlobalsPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    EncodingAnalysis analysis(moduleOp);
    if (failed(analysis.run())) {
      return;
    }
    for (auto globalOp : analysis.getGlobalOps()) {
      std::optional<ParameterWrapper> sourceGlobal =
          analysis.getSourceGlobal(globalOp);
      LLVM_DEBUG(llvm::dbgs() << "--GlobalOp:" << globalOp << "\n";);
      LLVM_DEBUG({
        llvm::dbgs() << "\tSource: ";
        if (sourceGlobal) {
          llvm::dbgs() << sourceGlobal.value();
        } else {
          llvm::dbgs() << "self";
        }
        llvm::dbgs() << "\n";
      });
      for (auto [consumerOp, encodings] :
           analysis.lookupGlobalConsumerLayouts(globalOp)) {
        LLVM_DEBUG({
          llvm::dbgs() << "\thas consumer: " << consumerOp << "\n";
          llvm::dbgs() << "\twhere the encoding is (";
          llvm::interleaveComma(encodings, llvm::dbgs(), [&](Attribute value) {
            value.print(llvm::dbgs());
          });
          llvm::dbgs() << ")\n";
        });
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
