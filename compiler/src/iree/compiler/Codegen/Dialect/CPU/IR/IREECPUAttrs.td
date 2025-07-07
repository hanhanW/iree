// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS
#define IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS

include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// CPU Specific Lowering Config Attributes
//===----------------------------------------------------------------------===//

// TODO(hanchung): Support pretty printer/parser for scalable vectors.
def IREECPU_LoweringConfigAttr :
    AttrDef<IREECPU_Dialect, "LoweringConfig", [
      DeclareAttrInterfaceMethods<IREECodegen_LoweringConfigAttrInterface, [
        "getWorkgroupTileSizes",
        "getTilingLevelSizes",
        "hasTilingLevel",
        "hasWorkgroupTilingLevel",
      ]>
    ]> {
  let mnemonic = "lowering_config";
  let summary = [{Drive lowering of an operation for gpu compilation.}];
  let description = [{
    CPU specific implementation of a lowering config. This carries just a
    dictionary attribute to store any relevant fields. This is the simplest
    form of a lowering config, offering flexibility at the cost of structure.
  }];

  let assemblyFormat = "`<` $config `>`";

  let parameters = (ins
    AttrParameter<"DictionaryAttr",
        "The configured fields, including tiling levels.">:$config
  );
}

//===----------------------------------------------------------------------===//
// Encoding Resolvers.
//===----------------------------------------------------------------------===//

def IREECPU_CPUEncodingResolverAttr :
    AttrDef<IREECPU_Dialect, "CPUEncodingResolver"> {
  let mnemonic = "cpu_encoding_resolver";
  let summary = [{The encoding layout attribute for CPU backends.}];
  let description = [{
    This attribute can implement any layout interface methods for encoding
    serialization and or materialization, e.g., Encoding::LayoutMaterializerAttr,
    Codegen::PackedLayoutMaterializerAttr, etc. They are implemented through external model
    mechanism See the implementation in compiler/Codegen/ExternalInterfaces/*.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    OptionalParameter<"DictionaryAttr", "Executable target configuration. It is "
    "expected to be used in a pass scope, but not the final IR output.">:$configuration
  );
}

def IREECPU_VMVXEncodingResolverAttr :
    AttrDef<IREECPU_Dialect, "VMVXEncodingResolver"> {
  let mnemonic = "vmvx_encoding_resolver";
  let summary = [{The encoding layout attribute for VMVX backend.}];
  let description = [{
    This attribute can implement any layout interface methods for encoding
    serialization and or materialization, e.g., Encoding::LayoutMaterializerAttr,
    Codegen::PackedLayoutMaterializerAttr, etc. They are implemented through external model
    mechanism See the implementation in compiler/Codegen/ExternalInterfaces/*.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    OptionalParameter<"DictionaryAttr", "Executable target configuration. It is "
    "expected to be used in a pass scope, but not the final IR output.">:$configuration
  );
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS
