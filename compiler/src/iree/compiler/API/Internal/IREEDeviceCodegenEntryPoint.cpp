// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Entry point for iree-device-codegen. Uses the IREE C API for
// initialization, CL parsing, and source parsing. Builds the device codegen
// pipeline programmatically with the session's target registry. Artifact
// extraction (workgroup_count .so, kernel binary, metadata.json) uses
// internal C++ APIs.

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetOptions.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "iree/compiler/embedding_api.h"
#include "iree/compiler/mlir_interop.h"
#include "iree/compiler/tool_entry_points_api.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

using namespace mlir;

namespace HAL = mlir::iree_compiler::IREE::HAL;

//===----------------------------------------------------------------------===//
// Workgroup count .so compilation
//===----------------------------------------------------------------------===//

// Creates a C-ABI wrapper: void @name(i64* grid, i64* args)
// around the original: {i64, i64, i64} @name_impl(i64 %0, ...)
static void createCABIWrapper(llvm::Module &M, llvm::Function *origFunc) {
  auto &ctx = M.getContext();
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  auto *ptrTy = llvm::PointerType::get(ctx, 0);
  auto *voidTy = llvm::Type::getVoidTy(ctx);

  std::string wrapperName = origFunc->getName().str();
  origFunc->setName(wrapperName + "_impl");
  origFunc->setLinkage(llvm::GlobalValue::InternalLinkage);

  auto *wrapperTy = llvm::FunctionType::get(voidTy, {ptrTy, ptrTy}, false);
  auto *wrapper = llvm::Function::Create(
      wrapperTy, llvm::GlobalValue::ExternalLinkage, wrapperName, &M);
  wrapper->setDSOLocal(true);

  auto *bb = llvm::BasicBlock::Create(ctx, "entry", wrapper);
  llvm::IRBuilder<> builder(bb);

  auto *gridPtr = wrapper->getArg(0);
  auto *argsPtr = wrapper->getArg(1);

  SmallVector<llvm::Value *> callArgs;
  for (unsigned i = 0; i < origFunc->arg_size(); ++i) {
    auto *gep = builder.CreateConstGEP1_64(i64Ty, argsPtr, i);
    callArgs.push_back(builder.CreateLoad(i64Ty, gep));
  }

  auto *result = builder.CreateCall(origFunc, callArgs);

  llvm::Type *retTy = origFunc->getReturnType();
  if (retTy->isStructTy()) {
    for (unsigned i = 0; i < 3; ++i) {
      auto *val = builder.CreateExtractValue(result, i);
      builder.CreateStore(val, builder.CreateConstGEP1_64(i64Ty, gridPtr, i));
    }
  } else if (retTy == i64Ty) {
    builder.CreateStore(result, gridPtr);
  }

  builder.CreateRetVoid();
}

static int compileWorkgroupCountToSO(ModuleOp module, StringRef outputDir) {
  SmallVector<func::FuncOp> wgCountFuncs;
  module.walk([&](func::FuncOp funcOp) {
    if (funcOp.getName().ends_with("_workgroup_count")) {
      wgCountFuncs.push_back(funcOp);
    }
  });

  if (wgCountFuncs.empty()) {
    llvm::errs() << "Warning: no workgroup_count functions found\n";
    return 0;
  }

  // Clone into a fresh module.
  MLIRContext *ctx = module.getContext();
  auto wgModule = ModuleOp::create(UnknownLoc::get(ctx), StringRef("wg_count"));
  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(wgModule.getBody());
  for (auto funcOp : wgCountFuncs) {
    auto clone = cast<func::FuncOp>(builder.clone(*funcOp));
    clone->removeAttr("workgroup_size");
    clone->removeAttr("subgroup_size");
  }

  // Lower to LLVM dialect.
  {
    PassManager pm(ctx);
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertIndexToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(pm.run(wgModule))) {
      llvm::errs() << "Error: LLVM lowering of workgroup_count failed\n";
      return 1;
    }
  }

  // Register translations and translate to LLVM IR.
  DialectRegistry translationRegistry;
  registerBuiltinDialectTranslation(translationRegistry);
  registerLLVMDialectTranslation(translationRegistry);
  ctx->appendDialectRegistry(translationRegistry);

  llvm::LLVMContext llvmContext;
  auto llvmModule =
      translateModuleToLLVMIR(wgModule.getOperation(), llvmContext, "wg_count");
  if (!llvmModule) {
    llvm::errs() << "Error: failed to translate workgroup_count to LLVM IR\n";
    return 1;
  }

  // Create C-ABI wrappers.
  SmallVector<llvm::Function *> origFuncs;
  for (auto &func : *llvmModule) {
    if (!func.isDeclaration() && func.getName().ends_with("_workgroup_count")) {
      origFuncs.push_back(&func);
    }
  }
  for (auto *func : origFuncs) {
    createCABIWrapper(*llvmModule, func);
  }

  // Compile to .o using host target.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::Triple triple(LLVM_HOST_TRIPLE);
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    llvm::errs() << "Error: cannot find host target: " << error << "\n";
    return 1;
  }

  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(triple, "generic", "", llvm::TargetOptions(),
                                  llvm::Reloc::PIC_));
  if (!targetMachine) {
    llvm::errs() << "Error: failed to create host target machine\n";
    return 1;
  }

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(triple);

  SmallString<0> objBuffer;
  {
    llvm::raw_svector_ostream objStream(objBuffer);
    llvm::legacy::PassManager codegenPasses;
    if (targetMachine->addPassesToEmitFile(codegenPasses, objStream, nullptr,
                                           llvm::CodeGenFileType::ObjectFile)) {
      llvm::errs() << "Error: target cannot emit object file\n";
      return 1;
    }
    codegenPasses.run(*llvmModule);
  }

  std::string objPath = (outputDir + "/workgroup_count.o").str();
  {
    std::error_code ec;
    llvm::raw_fd_ostream os(objPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "Error: cannot write " << objPath << "\n";
      return 1;
    }
    os.write(objBuffer.data(), objBuffer.size());
  }

  // Link to .so.
  std::string soPath = (outputDir + "/workgroup_count.so").str();
  std::string lldPath = iree_compiler::findTool(
      SmallVector<std::string>{"iree-lld", "lld", "ld.lld"});
  if (lldPath.empty()) {
    llvm::errs() << "Error: cannot find lld linker\n";
    return 1;
  }

  std::string linkCmd =
      lldPath + " -flavor gnu -shared -o " + soPath + " " + objPath;
  llvm::errs() << "Linking: " << linkCmd << "\n";
  if (std::system(linkCmd.c_str()) != 0) {
    llvm::errs() << "Error: linking failed\n";
    return 1;
  }

  llvm::sys::fs::remove(objPath);
  llvm::errs() << "Wrote: " << soPath << "\n";
  return 0;
}

//===----------------------------------------------------------------------===//
// Metadata + kernel binary extraction
//===----------------------------------------------------------------------===//

// Returns the kernel binary filename for the given backend.
static StringRef getKernelFilename(StringRef backendName) {
  if (backendName == "rocm") {
    return "kernel.hsaco";
  }
  if (backendName == "llvm-cpu") {
    return "kernel.so";
  }
  if (backendName == "vulkan-spirv") {
    return "kernel.spv";
  }
  return "kernel.bin";
}

static int writeArtifacts(ModuleOp module, StringRef outputDir,
                          StringRef backendName) {
  std::string kernelName;
  module.walk([&](HAL::ExecutableOp execOp) {
    kernelName = execOp.getSymName().str();
  });

  // Extract kernel binary.
  module.walk([&](HAL::ExecutableBinaryOp binaryOp) {
    auto dataAttr = dyn_cast<DenseIntElementsAttr>(binaryOp.getData());
    if (!dataAttr) {
      return;
    }
    std::string path = (outputDir + "/" + getKernelFilename(backendName)).str();
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) {
      return;
    }
    for (auto val : dataAttr.getValues<uint8_t>()) {
      fputc(val, f);
    }
    fclose(f);
    llvm::errs() << "Wrote: " << path << " (" << dataAttr.getNumElements()
                 << " bytes)\n";
  });

  // Extract workgroup_size/subgroup_size from extracted func, and
  // num_bindings/num_constants from the export op's layout.
  std::string wgSizeStr;
  int64_t subgroupSize = 0;
  int64_t numBindings = 0;
  int64_t numConstants = 0;
  module.walk([&](func::FuncOp funcOp) {
    if (!funcOp.getName().ends_with("_workgroup_count")) {
      return;
    }
    if (auto wgSize = funcOp->getAttrOfType<ArrayAttr>("workgroup_size")) {
      auto vals = wgSize.getValue();
      if (vals.size() >= 3) {
        auto get = [&](unsigned i) {
          return cast<IntegerAttr>(vals[i]).getInt();
        };
        char buf[64];
        snprintf(buf, sizeof(buf), "[%ld, %ld, %ld]", get(0), get(1), get(2));
        wgSizeStr = buf;
      }
    }
    if (auto sgSize = funcOp->getAttrOfType<IntegerAttr>("subgroup_size")) {
      subgroupSize = sgSize.getInt();
    }
  });
  // Get binding/constant counts from module attrs (set during pipeline).
  if (auto attr = module->getAttrOfType<IntegerAttr>(
          "iree.device_codegen.num_bindings")) {
    numBindings = attr.getInt();
  }
  if (auto attr = module->getAttrOfType<IntegerAttr>(
          "iree.device_codegen.num_constants")) {
    numConstants = attr.getInt();
  }

  // Write metadata.json.
  std::string metaPath = (outputDir + "/metadata.json").str();
  FILE *mf = fopen(metaPath.c_str(), "w");
  if (!mf) {
    llvm::errs() << "Error: cannot open " << metaPath << "\n";
    return 1;
  }
  fprintf(mf, "{\n");
  fprintf(mf, "  \"kernel_name\": \"%s\"", kernelName.c_str());
  fprintf(mf, ",\n  \"backend\": \"%s\"", backendName.str().c_str());
  fprintf(mf, ",\n  \"kernel_file\": \"%s\"",
          getKernelFilename(backendName).str().c_str());
  if (!wgSizeStr.empty()) {
    fprintf(mf, ",\n  \"workgroup_size\": %s", wgSizeStr.c_str());
  }
  if (subgroupSize > 0) {
    fprintf(mf, ",\n  \"subgroup_size\": %ld", subgroupSize);
  }
  fprintf(mf, ",\n  \"num_bindings\": %ld", numBindings);
  fprintf(mf, ",\n  \"num_constants\": %ld", numConstants);
  // Extract buffer alignment from native_vector_size in the target config.
  int64_t bufferAlignment = 64;  // Safe default for AVX-512.
  if (auto targetAttr =
          module->getAttrOfType<HAL::ExecutableTargetAttr>(
              "hal.executable.target")) {
    if (auto config = targetAttr.getConfiguration()) {
      if (auto nvs = config.getAs<IntegerAttr>("native_vector_size")) {
        bufferAlignment = nvs.getInt();
      }
    }
  }
  fprintf(mf, ",\n  \"buffer_alignment\": %ld", bufferAlignment);
  fprintf(mf, "\n}\n");
  fclose(mf);
  llvm::errs() << "Wrote: " << metaPath << "\n";
  return 0;
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

int ireeDeviceCodegenRunMain(int argc, char **argv) {
  // --- CL options ---
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));
  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));
  static llvm::cl::opt<std::string> outputDir(
      "output-dir",
      llvm::cl::desc("Extract artifacts (kernel binary, metadata.json, "
                     "workgroup_count.so) to directory"));
  static llvm::cl::opt<bool> noSerialize(
      "no-serialize",
      llvm::cl::desc("Skip serialization (codegen-only inspection)"),
      llvm::cl::init(false));

  // Inject backend-specific serialization flags based on argv scanning.
  // We do this before CL parsing since the flags need to be present at parse
  // time.
  SmallVector<const char *> augArgv(argv, argv + argc);
  {
    bool hasROCMContainerType = false;
    bool hasLLVMCPUContainerType = false;
    bool hasVulkanContainerType = false;
    bool isLLVMCPUBackend = false;
    bool isVulkanBackend = false;
    bool isNonROCMBackend = false;
    for (int i = 1; i < argc; ++i) {
      if (strncmp(argv[i], "--iree-rocm-container-type", 26) == 0) {
        hasROCMContainerType = true;
      }
      if (strncmp(argv[i], "--iree-llvmcpu-container-type", 29) == 0) {
        hasLLVMCPUContainerType = true;
      }
      if (strncmp(argv[i], "--iree-vulkan-container-type", 28) == 0) {
        hasVulkanContainerType = true;
      }
      if (strncmp(argv[i], "--iree-hal-target-backends=", 27) == 0) {
        if (strstr(argv[i], "llvm-cpu") != nullptr) {
          isLLVMCPUBackend = true;
        }
        if (strstr(argv[i], "vulkan-spirv") != nullptr) {
          isVulkanBackend = true;
        }
        if (strstr(argv[i], "rocm") == nullptr) {
          isNonROCMBackend = true;
        }
      }
    }
    // ROCM: force raw HSACO ELF output (not the default FlatBuffer).
    if (!hasROCMContainerType && !noSerialize && !isNonROCMBackend) {
      augArgv.push_back("--iree-rocm-container-type=hsaco");
    }
    // LLVMCPU: force raw .so output and system-native linking.
    if (isLLVMCPUBackend && !hasLLVMCPUContainerType) {
      augArgv.push_back("--iree-llvmcpu-container-type=raw");
      augArgv.push_back("--iree-llvmcpu-link-embedded=false");
    }
    // Vulkan: force raw SPIR-V output (not FlatBuffer).
    if (isVulkanBackend && !hasVulkanContainerType) {
      augArgv.push_back("--iree-vulkan-container-type=spirv");
    }
  }
  int augArgc = augArgv.size();
  const char **augArgvPtr = augArgv.data();

  // --- IREE C API initialization ---
  ireeCompilerGlobalInitialize();
  ireeCompilerSetupGlobalCL(augArgc, augArgvPtr,
                            "IREE device-only code generation tool\n",
                            /*installSignalHandlers=*/true);

  // --- Session, invocation, and source parsing ---
  auto *session = ireeCompilerSessionCreate();
  auto *inv = ireeCompilerInvocationCreate(session);
  ireeCompilerInvocationEnableConsoleDiagnostics(inv);

  iree_compiler_source_t *source = nullptr;
  auto *err =
      ireeCompilerSourceOpenFile(session, inputFilename.c_str(), &source);
  if (err) {
    llvm::errs() << "Error: " << ireeCompilerErrorGetMessage(err) << "\n";
    ireeCompilerErrorDestroy(err);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    ireeCompilerGlobalShutdown();
    return 1;
  }

  if (!ireeCompilerInvocationParseSource(inv, source)) {
    ireeCompilerSourceDestroy(source);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    ireeCompilerGlobalShutdown();
    return 1;
  }
  // NOTE: keep `source` alive — the invocation's SourceMgrDiagnosticHandler
  // holds a reference to its SourceMgr. Destroying it early causes crashes
  // when diagnostics (warnings/errors) are emitted during the pipeline.

  // --- Get session's target registry ---
  auto *registry = reinterpret_cast<const HAL::TargetRegistry *>(
      ireeCompilerSessionGetTargetRegistry(session));

  // --- Determine target backend ---
  // Priority: --iree-hal-target-backends > --iree-hal-target-device > default.
  std::string targetBackendName;
  {
    auto &targetOpts = HAL::TargetOptions::FromFlags::get();
    if (!targetOpts.legacyTargetBackends.empty()) {
      targetBackendName = targetOpts.legacyTargetBackends.front();
    } else if (!targetOpts.targetDevices.empty()) {
      // Map device name to backend name via the TargetBackend registry.
      // E.g., "hip" device -> "rocm" backend, "local" -> "llvm-cpu".
      StringRef deviceName = targetOpts.targetDevices.front();
      bool found = false;
      for (const auto &backendName : registry->getRegisteredTargetBackends()) {
        auto backend = registry->getTargetBackend(backendName);
        if (backend && backend->getLegacyDefaultDeviceID() == deviceName) {
          targetBackendName = backendName;
          found = true;
          break;
        }
      }
      if (!found) {
        llvm::errs() << "Error: no backend found for device '" << deviceName
                     << "'\n";
        ireeCompilerInvocationDestroy(inv);
        ireeCompilerSessionDestroy(session);
        ireeCompilerGlobalShutdown();
        return 1;
      }
    } else {
      targetBackendName = "rocm";
    }
  }

  // --- Steal module from invocation ---
  MlirOperation mlirOp = ireeCompilerInvocationExportStealModule(inv);
  auto moduleOp = cast<ModuleOp>(unwrap(mlirOp));
  MLIRContext *ctx = moduleOp.getContext();
  ctx->loadAllAvailableDialects();

  // --- Construct ExecutableTargetAttr and attach to module ---
  {
    auto backend = registry->getTargetBackend(targetBackendName);
    if (!backend) {
      llvm::errs() << "Error: unknown target backend '" << targetBackendName
                   << "'\n";
      moduleOp->erase();
      ireeCompilerInvocationDestroy(inv);
      ireeCompilerSessionDestroy(session);
      ireeCompilerGlobalShutdown();
      return 1;
    }
    std::string deviceID = backend->getLegacyDefaultDeviceID();
    SmallVector<HAL::ExecutableTargetAttr> targets;
    backend->getDefaultExecutableTargets(ctx, deviceID,
                                         DictionaryAttr::get(ctx), targets);
    if (targets.empty()) {
      llvm::errs() << "Error: target backend '" << targetBackendName
                   << "' returned no executable targets; check target flags "
                      "(e.g., --iree-rocm-target)\n";
      moduleOp->erase();
      ireeCompilerInvocationDestroy(inv);
      ireeCompilerSessionDestroy(session);
      ireeCompilerGlobalShutdown();
      return 1;
    }
    moduleOp->setAttr("hal.executable.target", targets.front());
  }

  // --- Phase 1: Convert func.func → hal.executable ---
  {
    PassManager pm(ctx);
    pm.enableVerifier();
    pm.addPass(iree_compiler::createConvertFuncToHALExecutablePass());
    if (failed(pm.run(moduleOp))) {
      llvm::errs() << "Error: ConvertFuncToHALExecutable failed\n";
      moduleOp->erase();
      ireeCompilerInvocationDestroy(inv);
      ireeCompilerSessionDestroy(session);
      ireeCompilerGlobalShutdown();
      return 1;
    }
  }

  // Extract binding/constant counts from the export op before the rest of
  // the pipeline consumes them. Store as module-level attributes for
  // metadata.json extraction later.
  moduleOp.walk([&](HAL::ExecutableExportOp exportOp) {
    if (auto layoutAttr = exportOp.getLayout()) {
      moduleOp->setAttr("iree.device_codegen.num_bindings",
                         IntegerAttr::get(IntegerType::get(ctx, 64),
                                          layoutAttr.getBindings().size()));
      moduleOp->setAttr("iree.device_codegen.num_constants",
                         IntegerAttr::get(IntegerType::get(ctx, 64),
                                          layoutAttr.getConstants()));
    }
  });

  // --- Phase 2: Codegen, translation, and serialization ---
  {
    PassManager pm(ctx);
    pm.enableVerifier();

    // Configure and translate executables with session registry.
    {
      auto &execPM = pm.nest<HAL::ExecutableOp>();

      iree_compiler::IREE::HAL::ConfigureExecutablesPassOptions configOpts;
      configOpts.targetRegistry = registry;
      execPM.addPass(
          iree_compiler::IREE::HAL::createConfigureExecutablesPass(configOpts));

      iree_compiler::IREE::HAL::TranslateAllExecutablesPassOptions
          translateOpts;
      translateOpts.targetRegistry = registry;
      execPM.addPass(
          iree_compiler::IREE::HAL::createTranslateAllExecutablesPass(
              translateOpts));
    }

    // Extract workgroup count as func.func.
    pm.addPass(iree_compiler::createExtractWorkgroupCountAsFuncPass());

    // Lower affine ops in func.func scope.
    if (failed(
            parsePassPipeline("func.func(lower-affine)", pm, llvm::errs()))) {
      llvm::errs() << "Error: failed to parse lower-affine pipeline\n";
      moduleOp->erase();
      ireeCompilerInvocationDestroy(inv);
      ireeCompilerSessionDestroy(session);
      ireeCompilerGlobalShutdown();
      return 1;
    }

    // Serialize executables (optional).
    if (!noSerialize) {
      iree_compiler::IREE::HAL::SerializeAllExecutablesPassOptions
          serializeOpts;
      serializeOpts.targetRegistry = registry;
      pm.nest<HAL::ExecutableOp>().addPass(
          iree_compiler::IREE::HAL::createSerializeAllExecutablesPass(
              serializeOpts));
    }

    if (failed(pm.run(moduleOp))) {
      llvm::errs() << "Error: codegen pipeline failed\n";
      moduleOp->erase();
      ireeCompilerInvocationDestroy(inv);
      ireeCompilerSessionDestroy(session);
      ireeCompilerGlobalShutdown();
      return 1;
    }
  }

  // --- Extract artifacts or print MLIR ---
  int result = 0;
  if (!outputDir.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(outputDir)) {
      llvm::errs() << "Error: cannot create " << outputDir << "\n";
      result = 1;
    }

    if (result == 0 && compileWorkgroupCountToSO(moduleOp, outputDir) != 0) {
      result = 1;
    }

    if (result == 0) {
      result = writeArtifacts(moduleOp, outputDir, targetBackendName);
    }
  } else {
    // Print MLIR to output file.
    std::string errorMessage;
    auto output = openOutputFile(outputFilename, &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
      result = 1;
    } else {
      moduleOp.print(output->os());
      output->keep();
    }
  }

  // --- Cleanup ---
  moduleOp->erase();
  ireeCompilerSourceDestroy(source);
  ireeCompilerInvocationDestroy(inv);
  ireeCompilerSessionDestroy(session);
  ireeCompilerGlobalShutdown();
  return result;
}
