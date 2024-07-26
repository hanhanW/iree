// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir::iree_compiler::IREE::Codegen {

FailureOr<ModuleOp>
IREECodegenDialect::getOrLoadTransformLibraryModule(std::string libraryPath) {
  // Acquire a lock on the map that will release once out of scope.
  std::lock_guard<std::mutex> guard(libraryMutex);

  auto loadedLibrary = libraryModules.find(libraryPath);
  if (loadedLibrary != libraryModules.end()) {
    // Check whether the library already failed to load.
    if (!(loadedLibrary->second) || !(*(loadedLibrary->second))) {
      return failure();
    }
    return *(loadedLibrary->second);
  }

  OwningOpRef<ModuleOp> mergedParsedLibraries;
  if (failed(transform::detail::assembleTransformLibraryFromPaths(
          getContext(), SmallVector<std::string>{libraryPath},
          mergedParsedLibraries))) {
    // We update the storage for the library regardless of whether parsing
    // succeeds so that other threads don't have to retry.
    OwningOpRef<ModuleOp> emptyLibrary;
    libraryModules[libraryPath] = std::move(emptyLibrary);
    return failure();
  }

  libraryModules[libraryPath] = std::move(mergedParsedLibraries);
  return *libraryModules[libraryPath];
}

FailureOr<::mlir::ModuleOp>
IREECodegenDialect::getOrLoadUkernelModule(std::string libraryPath) {
  // Acquire a lock on the map that will release once out of scope.
  std::lock_guard<std::mutex> guard(ukernelMutex);

  auto loadedLibrary = ukernelModules.find(libraryPath);
  if (loadedLibrary != ukernelModules.end()) {
    // Check whether the library already failed to load.
    if (!(loadedLibrary->second) || !(*(loadedLibrary->second))) {
      return failure();
    }
    return *(loadedLibrary->second);
  }

  std::string errorMessage;
  auto memoryBuffer = mlir::openInputFile(libraryPath, &errorMessage);
  if (!memoryBuffer) {
    return emitError(FileLineColLoc::get(
               StringAttr::get(getContext(), libraryPath), 0, 0))
           << "failed to open transform file: " << errorMessage;
  }
  // Tell sourceMgr about this buffer, the parser will pick it up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  auto moduleOp =
      OwningOpRef<ModuleOp>(parseSourceFile<ModuleOp>(sourceMgr, getContext()));
  if (!moduleOp) {
    // Failed to parse the transform module.
    // Don't need to emit an error here as the parsing should have already done
    // that.
    return failure();
  }
  if (failed(mlir::verify(*moduleOp))) {
    return failure();
  }

  ukernelModules[libraryPath] = std::move(moduleOp);
  return *ukernelModules[libraryPath];
}

} // namespace mlir::iree_compiler::IREE::Codegen
