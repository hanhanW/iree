// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <queue>

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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
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
  ParameterWrapper() = default;
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
/// Returns a stably sorted list of dialect interfaces of T for all dialects
/// used within the given module.
template <typename T>
SmallVector<const T *> gatherUsedDialectInterfaces(mlir::ModuleOp moduleOp) {
  SmallPtrSet<const T *, 4> resultSet;
  for (auto dialect : moduleOp.getContext()->getLoadedDialects()) {
    auto *dialectInterface = dialect->getRegisteredInterface<T>();
    if (!dialectInterface)
      continue;
    resultSet.insert(dialectInterface);
  }

  // NOTE: to ensure deterministic output we sort the result so that imports are
  // always added in a consistent order.
  auto results = llvm::to_vector_of<const T *>(resultSet);
  llvm::sort(
      results, +[](const T *a, const T *b) {
        return a->getDialect()->getNamespace().compare(
                   b->getDialect()->getNamespace()) < 0;
      });
  return results;
}
} // namespace

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

constexpr int kUnknown = -1;
using ParameterAndOpOperand = std::tuple<ParameterWrapper, int>;
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ParameterAndOpOperand &wrapper) {
  os << "(param: " << std::get<0>(wrapper);
  os << ", operandIdx: " << std::get<1>(wrapper) << ")";
  return os;
}

static const std::string
getLayoutSetAsStr(const DFX::PotentialValuesState<ParameterAndOpOperand> &state,
                  AsmState &asmState) {
  DenseSet<ParameterAndOpOperand, DenseMapInfo<ParameterAndOpOperand>>
      assumedSet = state.getAssumedSet();
  std::string str;
  llvm::raw_string_ostream sstream(str);
  sstream << "pvs: ";
  if (state.isValidState()) {
    sstream << "[";
    if (state.isUndefContained()) {
      sstream << "undef, ";
    }
    llvm::interleaveComma(state.getAssumedSet(), sstream,
                          [&](ParameterAndOpOperand item) {
                            sstream << "\n\t" << std::get<0>(item);
                            sstream << "idx: " << std::get<1>(item) << ")";
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

class DirectOpPVS
    : public DFX::StateWrapper<DFX::PotentialValuesState<ParameterAndOpOperand>,
                               DFX::OperationElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<ParameterAndOpOperand>,
                        DFX::OperationElement>;
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
    : public DFX::StateWrapper<DFX::PotentialValuesState<ParameterAndOpOperand>,
                               DFX::ValueElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<ParameterAndOpOperand>,
                        DFX::ValueElement>;
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
        for (auto [idx, operand] :
             llvm::enumerate(dispatchOp.getMixedOperands())) {
          auto &producerPVS = solver.getElementFor<DirectValueProducerPVS>(
              *this, Position::forValue(operand), DFX::Resolution::OPTIONAL);
          LLVM_DEBUG(producerPVS.getAsStr(solver.getAsmState()));
          SmallVector<ParameterAndOpOperand> newSet;
          for (const auto [globalOp, opOperand] :
               producerPVS.getState().getAssumedSet()) {
            (void)opOperand;
            newState.unionAssumed(ParameterAndOpOperand{globalOp, idx});
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
        ParameterAndOpOperand item = {ParameterWrapper(globalInfo->op),
                                      kUnknown};
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
        for (auto [globalOp, _] : sourceState.getState().getAssumedSet()) {
          ParameterAndOpOperand item = {globalOp, kUnknown};
          newState.unionAssumed(item);
        }
      })
      .Case<IREE::Stream::TensorConstantOp>([&](auto op) {
        if (auto attr =
                dyn_cast<IREE::Stream::NamedParameterAttr>(op.getValue())) {
          ParameterAndOpOperand item = {ParameterWrapper(attr.getKey()),
                                        kUnknown};
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

  SmallVector<ParameterAndOpOperand>
  lookupGlobalInputs(IREE::Stream::TensorDispatchOp dispatchOp);

  std::optional<ParameterWrapper> getSourceGlobal(ParameterWrapper op);

  bool updateEncoding(RewriterBase &rewriter, ParameterWrapper param,
                      Attribute newEncoding);

  SmallVector<IREE::Stream::TensorDispatchOp>
  lookupDispatchUsers(ParameterWrapper op);

  bool isDefinedWithEncodings(IREE::Util::GlobalOpInterface globalOp);

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
  SmallVector<IREE::Util::GlobalOpInterface> globalOps;
  DenseMap<ParameterWrapper, llvm::SmallSetVector<Item, 4>>
      globalConsumerLayouts;
  DenseMap<ParameterWrapper, SmallVector<IREE::Stream::TensorDispatchOp>>
      tensorDispatchUsersFromGlobal;
  DenseMap<IREE::Util::GlobalOpInterface, ParameterWrapper> sourceGlobalCache;
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

  // TODO: I guess propagating from load ops may be more efficient. I'm not sure
  // if it works though. It is just a prototype, so it will be revisited later.
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
      if (!sourceOp.op && !sourceOp.parameter.empty()) {
        sourceGlobalCache[globalOp] = sourceOp;
        continue;
      }
      if (sourceOp.op && !sourceGlobalCache.count(globalOp)) {
        sourceGlobalCache[globalOp] = sourceOp;
      }
    }
  }

  explorer.forEachFunctionLikeOp([&](FunctionOpInterface funcOp) {
    // Skip Initializer for now.
    // TODO: Check if we need it.
    if (isa<IREE::Util::InitializerOp>(funcOp)) {
      return;
    }
    funcOp.walk([&](IREE::Stream::TensorDispatchOp op) {
      SmallVector<ParameterAndOpOperand> globals = lookupGlobalInputs(op);
      LLVM_DEBUG({
        llvm::dbgs() << "TensorDispatchOp: " << op << "\n";
        for (auto global : globals) {
          llvm::dbgs() << "\t" << global << "\n";
        }
      });
      for (auto [global, operandIdx] : globals) {
        (void)operandIdx;
        tensorDispatchUsersFromGlobal[global].push_back(op);
      }
    });
  });

  return success();
}

SmallVector<ParameterAndOpOperand> EncodingAnalysis::lookupGlobalInputs(
    IREE::Stream::TensorDispatchOp dispatchOp) {
  llvm::SmallSetVector<ParameterAndOpOperand, 4> result;
  auto dispatchOpPVS = solver.getOrCreateElementFor<DirectOpPVS>(
      Position::forOperation(dispatchOp));
  for (auto [globalOp, operandIdx] : dispatchOpPVS.getAssumedSet()) {
    result.insert({globalOp, operandIdx});
  }
  return llvm::to_vector(result);
}

SmallVector<Item>
EncodingAnalysis::lookupGlobalConsumerLayouts(ParameterWrapper op) {
  return llvm::to_vector(globalConsumerLayouts[op]);
}

std::optional<ParameterWrapper>
EncodingAnalysis::getSourceGlobal(ParameterWrapper op) {
  if (!op.op) {
    // Parameter is always the source.
    return op;
  }
  if (!sourceGlobalCache.count(op.op)) {
    return std::nullopt;
  }
  return sourceGlobalCache.lookup(op.op);
  SmallVector<ParameterWrapper> result;
  auto globalPVS =
      solver.lookupElementFor<GlobalPVS>(Position::forOperation(op.op));
  LLVM_DEBUG(llvm::dbgs() << "Looking up source global for:"
                          << *op.op.getOperation() << "\n";);
  for (auto [sourceOp, encodings] : globalPVS->getAssumedSet()) {
    if (sourceOp == op) {
      LLVM_DEBUG(llvm::dbgs() << "Skip self as source" << "\n");
      continue;
    }
    if (encodings.empty()) {
      continue;
    }
    LLVM_DEBUG({ llvm::dbgs() << "Candidate:" << sourceOp << "\n"; });
    // if (result) {
    //   assert(false && "expect to be initialized once");
    //   return std::nullopt;
    // }
    result.push_back(sourceOp);
  }
  for (auto sourceOp : result) {
    if (!sourceOp.op && !sourceOp.parameter.empty()) {
      return sourceOp;
    }
  }
  for (auto sourceOp : result) {
    if (sourceOp.op) {
      return sourceOp;
    }
  }
  return std::nullopt;
}

bool EncodingAnalysis::isDefinedWithEncodings(
    IREE::Util::GlobalOpInterface globalOp) {
  if (!globalOp) {
    return false;
  }
  std::optional<ParameterWrapper> sourceGlobal =
      getSourceGlobal(ParameterWrapper(globalOp));
  assert(sourceGlobal && !sourceGlobal->op);
  for (auto [consumerOp, encodings] :
       lookupGlobalConsumerLayouts(*sourceGlobal)) {
    if (consumerOp.op != globalOp) {
      continue;
    }
    return !encodings.empty();
  }
  assert(false && "can't find the SSA chain");
  return true;
}

SmallVector<IREE::Stream::TensorDispatchOp>
EncodingAnalysis::lookupDispatchUsers(ParameterWrapper op) {
  return tensorDispatchUsersFromGlobal[op];
}

bool EncodingAnalysis::updateEncoding(RewriterBase &rewriter,
                                      ParameterWrapper param,
                                      Attribute newEncoding) {
  OpBuilder::InsertionGuard guard(rewriter);
  std::queue<Operation *> worklist;
  for (auto [consumerOp, encodingChain] : lookupGlobalConsumerLayouts(param)) {
    assert(consumerOp.op && "expects a global op, not parameter string");
    if (!isDefinedWithEncodings(consumerOp.op)) {
      LDBG() << "skipping updating " << *consumerOp.op.getOperation()
             << " because it is not defined with encodings";
      continue;
    }
    const Explorer::GlobalInfo *globalInfo =
        explorer.getGlobalInfo(consumerOp.op);
    for (IREE::Util::GlobalStoreOpInterface storeOp : globalInfo->getStores()) {
      worklist.push(storeOp);
    }
    while (!worklist.empty()) {
      Operation *op = worklist.front();
      worklist.pop();
      bool result =
          TypeSwitch<Operation *, bool>(op)
              .Case<IREE::Util::GlobalStoreOpInterface>([&](auto storeOp) {
                worklist.push(storeOp.getStoredGlobalValue().getDefiningOp());
                return true;
              })
              .Case<IREE::Stream::TensorEncodeOp>([&](auto encodeOp) {
                auto resultType =
                    dyn_cast<RankedTensorType>(encodeOp.getResultEncoding());
                if (!resultType) {
                  return false;
                }
                resultType = resultType.cloneWithEncoding(newEncoding);
                encodeOp.setResultEncoding(resultType);
                rewriter.setInsertionPoint(encodeOp);
                auto tensorSizeOfOp = IREE::Stream::TensorSizeOfOp::create(
                    rewriter, encodeOp.getLoc(), TypeAttr::get(resultType),
                    /*result_encoding_dims=*/ValueRange{},
                    encodeOp.getAffinityAttr());
                encodeOp.getResultSizeMutable().assign(
                    tensorSizeOfOp.getResult());
                return true;
              })
              .Case<IREE::Stream::AsyncTransferOp>([&](auto transferOp) {
                worklist.push(transferOp.getSource().getDefiningOp());
                return true;
              })
              .Case<IREE::Stream::TensorConstantOp>([&](auto constOp) {
                if (!param.op) {
                  auto attr = dyn_cast<IREE::Stream::NamedParameterAttr>(
                      constOp.getValue());
                  assert(param.parameter == attr.getKey());
                }
                return true;
              })
              .Default([](Operation *) { return false; });
      if (!result) {
        LDBG() << "failed to update " << *op;
        return false;
      }
    }
  }
  return true;
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
      LLVM_DEBUG({
        for (auto [consumerOp, encodings] :
             analysis.lookupGlobalConsumerLayouts(globalOp)) {
          llvm::dbgs() << "\thas consumer: " << consumerOp << "\n";
          llvm::dbgs() << "\twhere the encoding is (";
          llvm::interleaveComma(encodings, llvm::dbgs(), [&](Attribute value) {
            value.print(llvm::dbgs());
          });
          llvm::dbgs() << ")\n";
        }
      });
    }

    IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
    if (failed(affinityAnalysis.run())) {
      LLVM_DEBUG(llvm::dbgs() << "failed on running affinity analysis\n");
      return;
    }

    auto usedDialects = gatherUsedDialectInterfaces<
        IREE::Stream::AffinityAnalysisDialectInterface>(moduleOp);
    if (usedDialects.size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "expected only one dialect implementing "
                                 "AffinityAnalysisDialectInterface\n");
      return;
    }
    IREE::Stream::ResolveLayoutAttrFn resolveLayoutAttr =
        usedDialects[0]->makeLayoutAttrResolver(moduleOp);

    // Step 1. Select a common encoding for each global/parameter.
    // TODO: Extend the consumer global users with affinity:
    //         - Iterate all the source globals that have more than one
    //         consumer layouts (including its users because plain layout is
    //         also a layout):
    //         - If they have the same encoding resolver, query the resolver
    //         for encoding preference by passing a list of encodings.
    //         - If multiple encoding resolvers are present, randomly select a
    //         encoding. This may be the entry point of specializing encoding
    //         module for each target device (which is a list).
    //         TODO: Implement heuritic, maybe look up the number of uses.
    //         Note: it may be okay to have dup globals if they belong
    //         different targets. As long as they are come from data-tiling,
    //         they already need a buffer to hold the parameters. It may be
    //         true for non data-tiling cases as well.
    // Step 2.
    // - Create a new global with the selected encoding.
    // - Track all the consumers and replace the use chain with the new
    //   encoded global.
    // - Meanwhile, update the encodings for dispatch bindings. Two options:
    //   * Only replace the encodings in iree_tensor_ext load/store.
    //   * Insert unset_encoding + set_encoding right after/before load/store.
    //   Question: can this happen before encoding specialization? Do we need
    //   to duplicate the dispatches based on needs? I think the answer is
    //   yes, it can happen before encoding specialization, because it does
    //   not break the assumptions. Note: we can do it != it is the solution.
    IRRewriter rewriter(&getContext());
    for (ParameterWrapper globalOp : analysis.getGlobalOps()) {
      LDBG() << "resolving " << globalOp;
      std::optional<ParameterWrapper> sourceGlobal =
          analysis.getSourceGlobal(globalOp);
      if (!sourceGlobal || sourceGlobal->op) {
        LDBG() << "skip, because it is not a source global";
        continue;
      }
      SmallVector<IREE::Stream::TensorDispatchOp> users;
      SmallVector<ParameterWrapper> consumers;
      SetVector<Attribute> encodings;
      for (auto [consumerOp, consumerEncodings] :
           analysis.lookupGlobalConsumerLayouts(globalOp)) {
        if (!analysis.isDefinedWithEncodings(consumerOp.op)) {
          LDBG() << "skip, because it is not defined with encodings";
          continue;
        }
        users.append(analysis.lookupDispatchUsers(consumerOp));
        consumers.push_back(consumerOp);
        encodings.insert(consumerEncodings.begin(), consumerEncodings.end());
      }
      if (encodings.size() <= 1) {
        LDBG() << "skip, because it has a single layout";
        continue;
      }

      SmallVector<IREE::Stream::AffinityAndOpPair> queries;
      for (IREE::Stream::TensorDispatchOp dispatchOp : users) {
        LDBG() << "processing " << dispatchOp;
        for (auto [param, idx] : analysis.lookupGlobalInputs(dispatchOp)) {
          std::optional<ParameterWrapper> paramSourceGlobal =
              analysis.getSourceGlobal(param);
          if (!paramSourceGlobal ||
              !(paramSourceGlobal.value() == sourceGlobal.value())) {
            continue;
          }
          if (!analysis.isDefinedWithEncodings(param.op)) {
            LDBG() << "skip, because it is not defined with encodings";
            continue;
          }
          queries.emplace_back(dispatchOp.getAffinityAttr(), dispatchOp);

          Value operand = dispatchOp.getMixedOperands()[idx];
          SmallVector<IREE::Stream::AffinityAttr> affinityAttrs;
          if (!affinityAnalysis.tryLookupResourceAffinity(operand,
                                                          affinityAttrs)) {
            LDBG() << "failed to determine resource affinity for operand "
                   << operand;
            assert(false);
            return;
          }
          for (auto affinity : affinityAttrs) {
            queries.emplace_back(affinity, dispatchOp);
          }
        }
      }
      llvm::DenseMap<IREE::Stream::AffinityAndOpPair, SetVector<Attribute>>
          cachedLayoutAttrs;
      if (failed(resolveLayoutAttr(queries, cachedLayoutAttrs))) {
        LDBG() << "failed to resolve layouts for an query";
        assert(false);
        return;
      }
      SetVector<Attribute> resolvers;
      for (auto layoutAttrs : cachedLayoutAttrs.values()) {
        resolvers.insert(layoutAttrs.begin(), layoutAttrs.end());
      }
      LLVM_DEBUG({
        if (resolvers.size() == 1) {
          llvm::dbgs() << "Unique candidate:\n";
        } else {
          llvm::dbgs() << "Multiple candidates:\n";
        }
        for (auto i : resolvers) {
          llvm::dbgs() << i << "\n";
        }
      });

      // Now we get resolvers.
      if (resolvers.size() != 1) {
        LDBG() << "multiple resolvers are not supported, need a heuristic";
        return;
      }
      // TODO: Query from the resolver to decide the encoding.
      Attribute unifiedEncoding = encodings[0];
#if 0
      // Use identity for debugging.
      unifiedEncoding = IREE::Encoding::IdentityAttr::get(&getContext());
#endif
      LDBG() << "Selected encoding: " << unifiedEncoding;

      for (IREE::Stream::TensorDispatchOp dispatchOp : users) {
        LDBG() << "Updating " << dispatchOp;
        SmallVector<Type> newOperandEncodings =
            llvm::map_to_vector(dispatchOp.getOperandEncodings().getValue(),
                                [](Attribute typeAttr) -> Type {
                                  return cast<TypeAttr>(typeAttr).getValue();
                                });
        for (auto [param, idx] : analysis.lookupGlobalInputs(dispatchOp)) {
          std::optional<ParameterWrapper> paramSourceGlobal =
              analysis.getSourceGlobal(param);
          if (!paramSourceGlobal ||
              !(paramSourceGlobal.value() == sourceGlobal.value())) {
            continue;
          }
          if (!analysis.isDefinedWithEncodings(param.op)) {
            LDBG() << "skip, because it is not defined with encodings";
            continue;
          }
          newOperandEncodings[idx] =
              cast<RankedTensorType>(newOperandEncodings[idx])
                  .cloneWithEncoding(unifiedEncoding);
          LDBG() << "Updating new encoding for " << idx << "-th operand";
        }
        dispatchOp.setOperandEncodingsAttr(
            rewriter.getTypeArrayAttr(newOperandEncodings));
      }

      bool result =
          analysis.updateEncoding(rewriter, globalOp, unifiedEncoding);
      assert(result && "failed to update encoding ops");

      // TODO: Update executables. Currently it is not needed in the prototype,
      // because SpecializeEncoding pass handles the case.
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
