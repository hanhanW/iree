// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Entry point for iree-device-codegen. Lives inside the compiler shared
// library so it has access to all internal APIs (pass constructors, dialect
// registrations, plugin infrastructure).

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Target/init_targets.h"
#include "iree/compiler/Pipelines/Options.h"
#include "iree/compiler/PluginAPI/PluginManager.h"
#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Tools/init_llvmir_translations.h"
#include "iree/compiler/Tools/init_passes.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "iree/compiler/tool_entry_points_api.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
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

  auto *wrapperTy =
      llvm::FunctionType::get(voidTy, {ptrTy, ptrTy}, false);
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

static int compileWorkgroupCountToSO(ModuleOp module,
                                      StringRef outputDir) {
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
  auto llvmModule = translateModuleToLLVMIR(wgModule.getOperation(),
                                             llvmContext, "wg_count");
  if (!llvmModule) {
    llvm::errs() << "Error: failed to translate workgroup_count to LLVM IR\n";
    return 1;
  }

  // Create C-ABI wrappers.
  SmallVector<llvm::Function *> origFuncs;
  for (auto &func : *llvmModule) {
    if (!func.isDeclaration() &&
        func.getName().ends_with("_workgroup_count")) {
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
      target->createTargetMachine(triple, "generic", "",
                                  llvm::TargetOptions(),
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
// Metadata + HSACO extraction
//===----------------------------------------------------------------------===//

static int writeArtifacts(ModuleOp module, StringRef outputDir) {
  std::string kernelName;
  module.walk([&](HAL::ExecutableOp execOp) {
    kernelName = execOp.getSymName().str();
  });

  // Extract HSACO.
  module.walk([&](HAL::ExecutableBinaryOp binaryOp) {
    auto dataAttr = dyn_cast<DenseIntElementsAttr>(binaryOp.getData());
    if (!dataAttr) {
      return;
    }
    std::string path = (outputDir + "/kernel.hsaco").str();
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

  // Extract workgroup_size/subgroup_size from extracted func.
  std::string wgSizeStr;
  int64_t subgroupSize = 0;
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

  // Write metadata.json.
  std::string metaPath = (outputDir + "/metadata.json").str();
  FILE *mf = fopen(metaPath.c_str(), "w");
  if (!mf) {
    llvm::errs() << "Error: cannot open " << metaPath << "\n";
    return 1;
  }
  fprintf(mf, "{\n");
  fprintf(mf, "  \"kernel_name\": \"%s\"", kernelName.c_str());
  if (!wgSizeStr.empty()) {
    fprintf(mf, ",\n  \"workgroup_size\": %s", wgSizeStr.c_str());
  }
  if (subgroupSize > 0) {
    fprintf(mf, ",\n  \"subgroup_size\": %ld", subgroupSize);
  }
  fprintf(mf, "\n}\n");
  fclose(mf);
  llvm::errs() << "Wrote: " << metaPath << "\n";
  return 0;
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

int ireeDeviceCodegenRunMain(int argc, char **argv) {
  llvm::setBugReportMsg(
      "Please report issues to https://github.com/iree-org/iree/issues and "
      "include the crash backtrace.\n");

  // --- CL options ---
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));
  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"),
      llvm::cl::value_desc("filename"), llvm::cl::init("-"));
  static llvm::cl::opt<std::string> outputDir(
      "output-dir",
      llvm::cl::desc("Extract artifacts (kernel.hsaco, metadata.json, "
                      "workgroup_count.so) to directory"));
  static llvm::cl::opt<bool> noSerialize(
      "no-serialize",
      llvm::cl::desc("Skip HSACO serialization (codegen-only inspection)"),
      llvm::cl::init(false));

  // --- IREE initialization ---
  DialectRegistry registry;
  iree_compiler::registerAllDialects(registry);
  iree_compiler::registerAllPasses();
  iree_compiler::registerVMTargets();
  iree_compiler::registerLLVMIRTranslations(registry);

  // Plugin setup.
  iree_compiler::PluginManager pluginManager;
  if (!pluginManager.loadAvailablePlugins()) {
    llvm::errs() << "Error: failed to initialize IREE compiler plugins\n";
    return 1;
  }
  pluginManager.globalInitialize();
  pluginManager.registerPasses();
  pluginManager.registerGlobalDialects(registry);
  pluginManager.initializeCLI();

  // Register standard CL options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  iree_compiler::GlobalPipelineOptions::FromFlags::get();

  // Inject --iree-rocm-container-type=hsaco so serialization emits raw
  // HSACO ELF (not the default HIP FlatBuffer container).
  SmallVector<const char *> augArgv(argv, argv + argc);
  bool hasContainerType = false;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--iree-rocm-container-type", 26) == 0) {
      hasContainerType = true;
    }
  }
  if (!hasContainerType && !noSerialize) {
    augArgv.push_back("--iree-rocm-container-type=hsaco");
  }
  int augArgc = augArgv.size();
  const char **augArgvPtr = augArgv.data();

  // Parse.
  llvm::InitLLVM y(augArgc, augArgvPtr);
  llvm::cl::ParseCommandLineOptions(
      augArgc, augArgvPtr, "IREE device-only code generation tool\n");

  // Apply optimization defaults.
  {
    auto globalBinder = iree_compiler::OptionsBinder::global();
    globalBinder.applyOptimizationDefaults();
  }

  // Plugin session — registers target backends.
  auto localBinder = iree_compiler::OptionsBinder::local();
  auto &pluginManagerOptions =
      iree_compiler::PluginManagerOptions::FromFlags::get();
  iree_compiler::PluginManagerSession pluginSession(
      pluginManager, localBinder, pluginManagerOptions);
  if (failed(pluginSession.initializePlugins())) {
    llvm::errs() << "Error: failed to initialize plugins\n";
    return 1;
  }
  pluginSession.registerDialects(registry);

  HAL::TargetDeviceList targetDeviceList;
  pluginSession.populateHALTargetDevices(targetDeviceList);
  const_cast<HAL::TargetRegistry &>(HAL::TargetRegistry::getGlobal())
      .mergeFrom(targetDeviceList);
  HAL::TargetBackendList targetBackendList;
  pluginSession.populateHALTargetBackends(targetBackendList);
  const_cast<HAL::TargetRegistry &>(HAL::TargetRegistry::getGlobal())
      .mergeFrom(targetBackendList);

  // --- Create context and parse input ---
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string errorMessage;
  auto inputFile = openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(inputFile), llvm::SMLoc());

  ParserConfig parserConfig(&context);
  OwningOpRef<ModuleOp> moduleRef =
      parseSourceFile<ModuleOp>(*sourceMgr, parserConfig);
  if (!moduleRef) {
    llvm::errs() << "Error: failed to parse input\n";
    return 1;
  }

  // --- Build and run the pipeline programmatically ---
  {
    PassManager pm(&context);
    pm.enableVerifier();

    // Step 1: Wrap func.func into hal.executable structure.
    pm.addPass(iree_compiler::createConvertFuncToHALExecutablePass());

    // Step 2: GPU codegen (tiling, vectorization, lowering to LLVM+ROCDL).
    {
      auto &variantPM = pm.nest<HAL::ExecutableOp>()
                            .nest<HAL::ExecutableVariantOp>();
      iree_compiler::buildLLVMGPUCodegenConfigurationPassPipeline(variantPM);
      iree_compiler::buildLLVMGPUCodegenPassPipeline(
          variantPM, /*useROCM=*/true, /*preserveDebugInfo=*/false);
    }

    // Step 3: Extract workgroup count region as standalone func.
    pm.addPass(iree_compiler::createExtractWorkgroupCountAsFuncPass());

    // Step 4: Lower affine ops in the extracted func to arith.
    pm.addNestedPass<func::FuncOp>(createLowerAffinePass());

    // Step 5: Serialize to HSACO (unless --no-serialize).
    if (!noSerialize) {
      pm.nest<HAL::ExecutableOp>().addPass(
          HAL::createSerializeAllExecutablesPass());
    }

    if (failed(pm.run(*moduleRef))) {
      llvm::errs() << "Error: codegen pipeline failed\n";
      return 1;
    }
  }

  // --- Extract artifacts or print MLIR ---
  if (!outputDir.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(outputDir)) {
      llvm::errs() << "Error: cannot create " << outputDir << "\n";
      return 1;
    }

    if (compileWorkgroupCountToSO(*moduleRef, outputDir) != 0) {
      return 1;
    }

    return writeArtifacts(*moduleRef, outputDir);
  }

  // Print MLIR to output file.
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }
  moduleRef->print(output->os());
  output->keep();
  return 0;
}
