// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-codegen-convert-func-to-hal-executable"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTFUNCTOHALEXECUTABLEPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Metadata about a single argument parsed from the input func.func.
struct ArgInfo {
  enum Kind { Tensor, Index, Scalar };
  Kind kind;
  unsigned funcArgIndex;
  // For tensor args: the binding ordinal.
  int64_t bindingOrdinal = -1;
  // For tensor args: whether this is an output tensor.
  bool isOutput = false;
  // For index args: the push constant ordinal.
  int64_t pushConstantOrdinal = -1;
  // For index args tied to output tensors: the workload ordinal.
  int64_t workloadOrdinal = -1;
};

// Parses the func.func arguments according to the input format specification.
// Returns the argument info list, binding count, and push constant count.
static LogicalResult
parseArguments(func::FuncOp funcOp, SmallVectorImpl<ArgInfo> &argInfos,
               int64_t &bindingCount, int64_t &pushConstantCount,
               SmallVectorImpl<int64_t> &workloadOrdinalToPushConstant) {
  auto funcType = funcOp.getFunctionType();
  bindingCount = 0;
  pushConstantCount = 0;
  int64_t workloadOrdinalCount = 0;

  for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
    Type argType = funcType.getInput(i);
    ArgInfo info;
    info.funcArgIndex = i;

    if (auto tensorType = dyn_cast<RankedTensorType>(argType)) {
      info.kind = ArgInfo::Tensor;
      info.bindingOrdinal = bindingCount++;
      // Check for iree.abi.output attribute.
      if (funcOp.getArgAttr(i, "iree.abi.output")) {
        info.isOutput = true;
      }
    } else if (argType.isIndex()) {
      info.kind = ArgInfo::Index;
      info.pushConstantOrdinal = pushConstantCount++;
    } else if (argType.isIntOrFloat()) {
      info.kind = ArgInfo::Scalar;
      info.pushConstantOrdinal = pushConstantCount++;
    } else {
      return funcOp.emitError("unsupported argument type at index ")
             << i << ": " << argType;
    }

    argInfos.push_back(info);
  }

  // Assign workload ordinals to ALL index args. The workgroup count
  // computation may trace through any dynamic dimension (input or output),
  // so all index args must be available as workload ordinals. This matches
  // IREE's standard flow where all captured index values become workloads.
  for (auto &info : argInfos) {
    if (info.kind == ArgInfo::Index) {
      info.workloadOrdinal = workloadOrdinalCount++;
      workloadOrdinalToPushConstant.push_back(info.pushConstantOrdinal);
    }
  }

  return success();
}

// Creates the #hal.pipeline.layout attribute from parsed arg info.
static IREE::HAL::PipelineLayoutAttr
buildPipelineLayout(MLIRContext *ctx, ArrayRef<ArgInfo> argInfos,
                    int64_t bindingCount, int64_t pushConstantCount) {
  SmallVector<IREE::HAL::PipelineBindingAttr> bindings;
  for (auto &info : argInfos) {
    if (info.kind != ArgInfo::Tensor)
      continue;
    auto flags = info.isOutput ? IREE::HAL::DescriptorFlags::None
                               : IREE::HAL::DescriptorFlags::ReadOnly;
    bindings.push_back(IREE::HAL::PipelineBindingAttr::get(
        ctx, IREE::HAL::DescriptorType::StorageBuffer, flags));
  }
  return IREE::HAL::PipelineLayoutAttr::get(
      ctx, bindings, pushConstantCount, IREE::HAL::PipelineLayoutFlags::None);
}

// For each tensor arg, determines which subsequent index args represent its
// dynamic dimensions. Returns a map from tensor funcArgIndex to a list of
// index funcArgIndices.
static DenseMap<unsigned, SmallVector<unsigned>>
buildTensorDimMap(func::FuncOp funcOp, ArrayRef<ArgInfo> argInfos) {
  DenseMap<unsigned, SmallVector<unsigned>> dimMap;

  for (unsigned i = 0; i < argInfos.size(); ++i) {
    if (argInfos[i].kind != ArgInfo::Tensor)
      continue;
    auto tensorType =
        cast<RankedTensorType>(funcOp.getFunctionType().getInput(i));
    int64_t numDynDims = tensorType.getNumDynamicDims();
    SmallVector<unsigned> dims;
    // The next numDynDims args should be index-typed dim sizes.
    unsigned nextIdx = i + 1;
    for (int64_t d = 0; d < numDynDims && nextIdx < argInfos.size();
         ++nextIdx) {
      if (argInfos[nextIdx].kind == ArgInfo::Index) {
        dims.push_back(nextIdx);
        ++d;
      } else {
        break;
      }
    }
    dimMap[i] = std::move(dims);
  }
  return dimMap;
}

// Creates the inner func.func with HAL interface ops replacing the original
// tensor/index arguments.
static func::FuncOp buildInnerFunc(
    OpBuilder &builder, Location loc, StringRef name,
    func::FuncOp sourceFuncOp, ArrayRef<ArgInfo> argInfos,
    IREE::HAL::PipelineLayoutAttr layoutAttr,
    const DenseMap<unsigned, SmallVector<unsigned>> &tensorDimMap) {
  auto funcType = FunctionType::get(builder.getContext(), {}, {});
  auto newFuncOp = func::FuncOp::create(builder, loc, name, funcType);

  // Clone the body from the source function.
  IRMapping mapper;
  Block *entryBlock = newFuncOp.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entryBlock);

  Value c0 = arith::ConstantIndexOp::create(entryBuilder, loc, 0);

  // First pass: create push constant loads for all index/scalar args.
  DenseMap<unsigned, Value> argValues;
  for (auto &info : argInfos) {
    if (info.kind == ArgInfo::Index) {
      // Load push constant as i32, then cast to index.
      auto loadOp = IREE::HAL::InterfaceConstantLoadOp::create(
          entryBuilder, loc, entryBuilder.getI32Type(), layoutAttr,
          entryBuilder.getIndexAttr(info.pushConstantOrdinal),
          /*alignment=*/IntegerAttr{}, /*values=*/ArrayAttr{});
      Value indexVal = arith::IndexCastUIOp::create(
          entryBuilder, loc, entryBuilder.getIndexType(), loadOp);

      // If this is a workload ordinal, wrap it.
      if (info.workloadOrdinal >= 0) {
        indexVal = IREE::TensorExt::DispatchWorkloadOrdinalOp::create(
            entryBuilder, loc, indexVal,
            entryBuilder.getIndexAttr(info.workloadOrdinal));
      }

      argValues[info.funcArgIndex] = indexVal;
    } else if (info.kind == ArgInfo::Scalar) {
      auto loadOp = IREE::HAL::InterfaceConstantLoadOp::create(
          entryBuilder, loc, entryBuilder.getI32Type(), layoutAttr,
          entryBuilder.getIndexAttr(info.pushConstantOrdinal),
          /*alignment=*/IntegerAttr{}, /*values=*/ArrayAttr{});
      Type origType =
          sourceFuncOp.getFunctionType().getInput(info.funcArgIndex);
      Value val = loadOp.getResult();
      // Cast from i32 to the original scalar type if needed.
      if (origType != entryBuilder.getI32Type()) {
        if (origType.isIntOrIndex()) {
          val = arith::IndexCastUIOp::create(entryBuilder, loc, origType, val);
        } else {
          val = arith::BitcastOp::create(entryBuilder, loc, origType, val);
        }
      }
      argValues[info.funcArgIndex] = val;
    }
  }

  // Second pass: create binding subspans and tensor loads for tensor args.
  for (auto &info : argInfos) {
    if (info.kind != ArgInfo::Tensor)
      continue;

    auto tensorType = cast<RankedTensorType>(
        sourceFuncOp.getFunctionType().getInput(info.funcArgIndex));
    auto access = info.isOutput
                      ? IREE::TensorExt::TensorAccess::WriteOnly
                      : IREE::TensorExt::TensorAccess::ReadOnly;
    auto dispatchTensorType =
        IREE::TensorExt::DispatchTensorType::get(access, tensorType);

    // Gather dynamic dim values for this tensor.
    SmallVector<Value> dynamicDims;
    auto it = tensorDimMap.find(info.funcArgIndex);
    if (it != tensorDimMap.end()) {
      for (unsigned dimArgIdx : it->second) {
        dynamicDims.push_back(argValues[dimArgIdx]);
      }
    }

    auto flags = info.isOutput ? IREE::HAL::DescriptorFlags::None
                               : IREE::HAL::DescriptorFlags::ReadOnly;
    auto bindingOp = IREE::HAL::InterfaceBindingSubspanOp::create(
        entryBuilder, loc, dispatchTensorType, layoutAttr,
        APInt(64, info.bindingOrdinal), /*byte_offset=*/c0, dynamicDims,
        /*alignment=*/entryBuilder.getIndexAttr(64), flags);

    if (!info.isOutput) {
      // For input tensors: load the dispatch tensor.
      SmallVector<OpFoldResult> offsets(tensorType.getRank(),
                                        entryBuilder.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(tensorType.getRank(),
                                        entryBuilder.getIndexAttr(1));
      SmallVector<OpFoldResult> sizes;
      unsigned dynDimIdx = 0;
      for (int64_t i = 0; i < tensorType.getRank(); ++i) {
        if (tensorType.isDynamicDim(i)) {
          sizes.push_back(dynamicDims[dynDimIdx++]);
        } else {
          sizes.push_back(entryBuilder.getIndexAttr(tensorType.getDimSize(i)));
        }
      }
      auto loadOp = IREE::TensorExt::DispatchTensorLoadOp::create(
          entryBuilder, loc, tensorType, bindingOp, dynamicDims, offsets, sizes,
          strides);
      argValues[info.funcArgIndex] = loadOp;
    } else {
      // For output tensors: store reference for later.
      argValues[info.funcArgIndex] = bindingOp;
    }
  }

  // Clone the body operations from the source function.
  // Map source block arguments to HAL interface values.
  Block *sourceEntry = &sourceFuncOp.getBody().front();
  for (unsigned i = 0; i < sourceEntry->getNumArguments(); ++i) {
    BlockArgument sourceArg = sourceEntry->getArgument(i);
    if (argValues.count(i)) {
      mapper.map(sourceArg, argValues[i]);
    }
  }

  for (auto &op : sourceEntry->without_terminator()) {
    entryBuilder.clone(op, mapper);
  }

  // Handle the return: store the result to the output binding.
  auto returnOp = cast<func::ReturnOp>(sourceEntry->getTerminator());
  for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
    Value result = mapper.lookupOrDefault(returnOp.getOperand(i));

    // Find the output binding for this result.
    for (auto &info : argInfos) {
      if (info.kind != ArgInfo::Tensor || !info.isOutput)
        continue;
      // Match via iree.abi.output index.
      auto outputAttr =
          sourceFuncOp.getArgAttr(info.funcArgIndex, "iree.abi.output");
      auto outputIdx = cast<IntegerAttr>(outputAttr).getInt();
      if (outputIdx != static_cast<int64_t>(i))
        continue;

      Value outputBinding = argValues[info.funcArgIndex];
      auto tensorType = cast<RankedTensorType>(
          sourceFuncOp.getFunctionType().getInput(info.funcArgIndex));

      SmallVector<Value> dynamicDims;
      auto dimIt = tensorDimMap.find(info.funcArgIndex);
      if (dimIt != tensorDimMap.end()) {
        for (unsigned dimArgIdx : dimIt->second) {
          dynamicDims.push_back(argValues[dimArgIdx]);
        }
      }

      SmallVector<OpFoldResult> offsets(tensorType.getRank(),
                                        entryBuilder.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(tensorType.getRank(),
                                        entryBuilder.getIndexAttr(1));
      SmallVector<OpFoldResult> sizes;
      unsigned dynDimIdx = 0;
      for (int64_t j = 0; j < tensorType.getRank(); ++j) {
        if (tensorType.isDynamicDim(j)) {
          sizes.push_back(dynamicDims[dynDimIdx++]);
        } else {
          sizes.push_back(
              entryBuilder.getIndexAttr(tensorType.getDimSize(j)));
        }
      }

      IREE::TensorExt::DispatchTensorStoreOp::create(
          entryBuilder, loc, result, outputBinding, dynamicDims, offsets, sizes,
          strides);
      break;
    }
  }

  func::ReturnOp::create(entryBuilder, loc);
  return newFuncOp;
}

struct ConvertFuncToHALExecutablePass final
    : impl::ConvertFuncToHALExecutablePassBase<ConvertFuncToHALExecutablePass> {
  using Base::Base;

  void runOnOperation() override {
    // Find func.func ops to convert.
    SmallVector<func::FuncOp> funcsToConvert;
    getOperation().walk([&](func::FuncOp funcOp) {
      // Only convert public functions with tensor arguments.
      if (!funcOp.isPublic())
        return;
      bool hasTensorArg = false;
      for (Type argType : funcOp.getFunctionType().getInputs()) {
        if (isa<RankedTensorType>(argType)) {
          hasTensorArg = true;
          break;
        }
      }
      if (hasTensorArg)
        funcsToConvert.push_back(funcOp);
    });

    if (funcsToConvert.empty())
      return;

    for (auto funcOp : funcsToConvert) {
      if (failed(convertFunc(funcOp)))
        return signalPassFailure();
    }
  }

  LogicalResult convertFunc(func::FuncOp funcOp) {
    Location loc = funcOp.getLoc();
    MLIRContext *ctx = &getContext();
    OpBuilder moduleBuilder(ctx);
    moduleBuilder.setInsertionPoint(funcOp);

    // Parse the argument structure.
    SmallVector<ArgInfo> argInfos;
    int64_t bindingCount = 0, pushConstantCount = 0;
    SmallVector<int64_t> workloadOrdinalToPushConstant;
    if (failed(parseArguments(funcOp, argInfos, bindingCount,
                              pushConstantCount,
                              workloadOrdinalToPushConstant)))
      return failure();

    // Build the pipeline layout.
    auto layoutAttr =
        buildPipelineLayout(ctx, argInfos, bindingCount, pushConstantCount);

    // Build tensor dim map.
    auto tensorDimMap = buildTensorDimMap(funcOp, argInfos);

    // Create target attribute with full GPU target info if available.
    IREE::HAL::ExecutableTargetAttr targetAttr;
    {
      SmallVector<NamedAttribute> config;
      if (auto gpuTarget = getCLGPUTarget(ctx)) {
        addConfigGPUTarget(ctx, gpuTarget, config);
        config.emplace_back(StringAttr::get(ctx, "abi"),
                            StringAttr::get(ctx, "hip"));
      }
      targetAttr = IREE::HAL::ExecutableTargetAttr::get(
          ctx, StringAttr::get(ctx, targetBackend),
          StringAttr::get(ctx, targetFormat),
          DictionaryAttr::get(ctx, config));
    }

    // Create the hal.executable.
    StringRef funcName = funcOp.getName();
    auto executableOp = IREE::HAL::ExecutableOp::create(
        moduleBuilder, loc, funcName);
    executableOp.setPublic();

    // Create the hal.executable.variant.
    OpBuilder execBuilder(executableOp.getBody());
    auto variantOp = IREE::HAL::ExecutableVariantOp::create(
        execBuilder, loc, targetAttr.getSymbolNameFragment(), targetAttr);

    // Create the hal.executable.export with workgroup count region.
    OpBuilder variantBuilder(variantOp.getBody());

    auto exportOp = IREE::HAL::ExecutableExportOp::create(
        variantBuilder, loc, moduleBuilder.getStringAttr(funcName),
        moduleBuilder.getIndexAttr(0), layoutAttr,
        /*workgroup_size=*/ArrayAttr{},
        /*subgroup_size=*/IntegerAttr{},
        /*workgroup_local_memory=*/IntegerAttr{});

    // Build the workgroup count region.
    // count(%device: !hal.device, %workload0: index, ...) -> (x, y, z)
    {
      Region &countRegion = exportOp.getWorkgroupCount();
      Block *countBlock = new Block();
      countRegion.push_back(countBlock);
      // First argument: !hal.device
      countBlock->addArgument(moduleBuilder.getType<IREE::HAL::DeviceType>(),
                              loc);
      // Workload arguments (one per output dynamic dim).
      SmallVector<Value> workloadArgs;
      for (size_t i = 0; i < workloadOrdinalToPushConstant.size(); ++i) {
        countBlock->addArgument(moduleBuilder.getIndexType(), loc);
        workloadArgs.push_back(
            countBlock->getArgument(countBlock->getNumArguments() - 1));
      }
      OpBuilder countBuilder = OpBuilder::atBlockEnd(countBlock);
      Operation *countOp =
          IREE::TensorExt::DispatchWorkgroupCountFromSliceOp::create(
              countBuilder, loc, workloadArgs);
      IREE::HAL::ReturnOp::create(countBuilder, loc,
                                   countOp->getResults());
    }

    // Create the inner builtin.module inside the variant (before terminator).
    {
      OpBuilder beforeTermBuilder = OpBuilder::atBlockTerminator(
          &variantOp.getBlock());
      ModuleOp::create(beforeTermBuilder, loc);
    }
    auto innerModuleOp = variantOp.getInnerModule();

    OpBuilder innerBuilder = OpBuilder::atBlockBegin(
        innerModuleOp.getBody());
    buildInnerFunc(innerBuilder, loc, funcName, funcOp,
                   argInfos, layoutAttr, tensorDimMap);

    // Remove the original func.func.
    funcOp.erase();

    return success();
  }
};

} // namespace
} // namespace mlir::iree_compiler
