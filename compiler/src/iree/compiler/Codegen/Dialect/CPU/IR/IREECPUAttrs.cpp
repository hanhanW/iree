// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp.inc"

namespace mlir::iree_compiler::IREE::CPU {

//===----------------------------------------------------------------------===//
// CPU Specific Lowering Config Attributes
//===----------------------------------------------------------------------===//

constexpr StringLiteral kDistributionConfigKey = "distribution";
constexpr StringLiteral kCacheParallelConfigKey = "cache_parallel";
constexpr StringLiteral kCacheReductionConfigKey = "cache_reduction";
constexpr StringLiteral kVectorCommonParallelConfigKey =
    "vector_common_parallel";
constexpr StringLiteral kVectorReductionConfigKey = "vector_reduction";
constexpr StringLiteral kVectorInnerParallelConfigKey = "vector_inner_parallel";

/// Returns the entry key for the config in CPU::LoweringConfigAttr. Returns
/// null if `level` is invalid.
StringRef getTilingLevelName(TilingLevel level) {
  switch (level) {
  case DistributionTiles:
    return kDistributionConfigKey;
  case CacheParallelTiles:
    return kCacheParallelConfigKey;
  case CacheReductionTiles:
    return kCacheReductionConfigKey;
  case VectorCommonParallelTiles:
    return kVectorCommonParallelConfigKey;
  case VectorReductionTiles:
    return kVectorReductionConfigKey;
  case VectorInnerParallelTiles:
    return kVectorInnerParallelConfigKey;
  case MaxNumTileLevels:
  case InvalidLevel:
  default:
    return StringRef();
  }
}

static SmallVector<int64_t> getTileSizes(DictionaryAttr config,
                                         CPU::TilingLevel level) {
  return extractFromIntegerArrayAttr<int64_t>(
      config.getAs<ArrayAttr>(getTilingLevelName(level)));
}

SmallVector<int64_t> LoweringConfigAttr::getWorkgroupTileSizes() const {
  return getTileSizes(getConfig(), CPU::DistributionTiles);
}

SmallVector<OpFoldResult>
LoweringConfigAttr::getTilingLevelSizes(OpBuilder &builder, unsigned level,
                                        Operation *op) const {
  assert(level < llvm::to_underlying(TilingLevel::MaxNumTileLevels) &&
         "invalid level");
  return llvm::map_to_vector(
      getTileSizes(getConfig(), static_cast<TilingLevel>(level)),
      [&](int64_t t) -> OpFoldResult { return builder.getIndexAttr(t); });
}

bool LoweringConfigAttr::hasTilingLevel(unsigned level) const {
  return getConfig().contains(
      getTilingLevelName(static_cast<TilingLevel>(level)));
}

bool LoweringConfigAttr::hasWorkgroupTilingLevel() const {
  return !getWorkgroupTileSizes().empty();
}

//===----------------------------------------------------------------------===//
// Attribute Registration
//===----------------------------------------------------------------------===//

void IREECPUDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::CPU
