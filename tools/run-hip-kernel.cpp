// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Generic HIP kernel launcher. Zero IREE dependencies.
//
// Loads an HSACO binary, allocates buffers from --input descriptions,
// launches the kernel, and compares output against --expected_output.
//
// Usage (with artifacts directory):
//   run-hip-kernel --artifacts=dir \
//       --input=4x128x256xf32 --input=4x256x512xf32 \
//       --input=4x128xf32 --input=4x128x512xf32 \
//       --expected_output=@expected.npy \
//       [--push-constants=val0,val1,...] [--atol=1e-5] [--rtol=1e-5]
//
// Usage (explicit paths):
//   run-hip-kernel --hsaco=kernel.hsaco --kernel=name \
//       --workgroup=Bx,By,Bz --grid=Gx,Gy,Gz \
//       --input=4x128x256xf32 --expected_output=@expected.npy
//
// Input format: [shape]x[type][=source]
//   4x128x256xf32         Random data (seed 42)
//   4x128x256xf32=0       Zeros
//   4x128x256xf32=1       Ones
//   4x128x256xf32=@f.npy  Load from numpy file
//
// The last --input is treated as the output buffer (read back after launch).

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                    \
  do {                                                     \
    hipError_t err = (expr);                               \
    if (err != hipSuccess) {                               \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", err, \
              hipGetErrorString(err), __FILE__, __LINE__); \
      exit(1);                                             \
    }                                                      \
  } while (0)

// A buffer described by shape + element type.
struct Buffer {
  std::vector<int64_t> shape;
  std::string dtype;   // "f32", "f16", "i32", etc.
  std::string source;  // "", "0", "1", "@file.npy"
  std::vector<float> hostData;
  void *devicePtr = nullptr;

  size_t numElements() const {
    size_t n = 1;
    for (auto d : shape) {
      n *= d;
    }
    return n;
  }

  size_t elementSize() const {
    if (dtype == "f32" || dtype == "i32") return 4;
    if (dtype == "f16" || dtype == "bf16") return 2;
    if (dtype == "i8") return 1;
    if (dtype == "f64" || dtype == "i64") return 8;
    return 4;
  }

  size_t sizeBytes() const { return numElements() * elementSize(); }
};

// Parse "4x128x256xf32=@file.npy" into a Buffer.
static Buffer parseBuffer(const std::string &desc) {
  Buffer buf;
  std::string rest = desc;

  // Split off =source if present.
  auto eq = rest.find('=');
  if (eq != std::string::npos) {
    buf.source = rest.substr(eq + 1);
    rest = rest.substr(0, eq);
  }

  // Parse dims: 4x128x256xf32
  // The last segment without digits is the dtype.
  size_t pos = 0;
  while (pos < rest.size()) {
    auto xpos = rest.find('x', pos);
    std::string tok = (xpos == std::string::npos)
                          ? rest.substr(pos)
                          : rest.substr(pos, xpos - pos);
    // If it starts with a letter, it's the dtype.
    if (!tok.empty() && (tok[0] < '0' || tok[0] > '9')) {
      buf.dtype = tok;
      break;
    }
    buf.shape.push_back(std::stol(tok));
    if (xpos == std::string::npos) break;
    pos = xpos + 1;
  }
  if (buf.dtype.empty()) buf.dtype = "f32";
  return buf;
}

// Load numpy .npy file (float32 only for now).
static bool loadNpy(const std::string &path, std::vector<float> &data) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;

  // Read header: magic \x93NUMPY + version + header_len + header string.
  char magic[6];
  if (fread(magic, 1, 6, f) != 6) {
    fclose(f);
    return false;
  }
  uint8_t major, minor;
  if (fread(&major, 1, 1, f) != 1) {
    fclose(f);
    return false;
  }
  if (fread(&minor, 1, 1, f) != 1) {
    fclose(f);
    return false;
  }

  uint32_t headerLen = 0;
  if (major >= 2) {
    if (fread(&headerLen, 4, 1, f) != 1) {
      fclose(f);
      return false;
    }
  } else {
    uint16_t hl16;
    if (fread(&hl16, 2, 1, f) != 1) {
      fclose(f);
      return false;
    }
    headerLen = hl16;
  }

  // Skip header string.
  fseek(f, headerLen, SEEK_CUR);

  // Read raw data.
  long dataStart = ftell(f);
  fseek(f, 0, SEEK_END);
  long dataSize = ftell(f) - dataStart;
  fseek(f, dataStart, SEEK_SET);

  data.resize(dataSize / sizeof(float));
  size_t read = fread(data.data(), sizeof(float), data.size(), f);
  fclose(f);
  return read == data.size();
}

// Save numpy .npy file (float32).
static bool saveNpy(const std::string &path, const float *data, size_t n,
                    const std::vector<int64_t> &shape) {
  FILE *f = fopen(path.c_str(), "wb");
  if (!f) return false;

  // Build header.
  std::string shapeStr = "(";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i > 0) shapeStr += ", ";
    shapeStr += std::to_string(shape[i]);
  }
  if (shape.size() == 1) shapeStr += ",";
  shapeStr += ")";
  std::string header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': " + shapeStr + ", }";
  // Pad to 64-byte alignment.
  size_t totalLen =
      10 + header.size() + 1;  // magic+version+headerlen+header+\n
  size_t pad = (64 - (totalLen % 64)) % 64;
  header.append(pad, ' ');
  header += '\n';

  uint16_t headerLen = header.size();
  fwrite("\x93NUMPY", 1, 6, f);
  uint8_t ver[2] = {1, 0};
  fwrite(ver, 1, 2, f);
  fwrite(&headerLen, 2, 1, f);
  fwrite(header.data(), 1, header.size(), f);
  fwrite(data, sizeof(float), n, f);
  fclose(f);
  return true;
}

static void initBuffer(Buffer &buf) {
  size_t n = buf.numElements();
  buf.hostData.resize(n);
  if (buf.source.empty()) {
    // Random data.
    for (size_t i = 0; i < n; i++) {
      buf.hostData[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }
  } else if (buf.source == "0") {
    std::fill(buf.hostData.begin(), buf.hostData.end(), 0.0f);
  } else if (buf.source == "1") {
    std::fill(buf.hostData.begin(), buf.hostData.end(), 1.0f);
  } else if (buf.source[0] == '@') {
    if (!loadNpy(buf.source.substr(1), buf.hostData)) {
      fprintf(stderr, "Error: cannot load %s\n", buf.source.c_str() + 1);
      exit(1);
    }
  } else {
    // Splat value.
    float val = std::stof(buf.source);
    std::fill(buf.hostData.begin(), buf.hostData.end(), val);
  }
}

static int verify(const float *got, const float *ref, size_t n, float atol,
                  float rtol) {
  int errors = 0;
  float maxAbsErr = 0.0f;
  float maxRelErr = 0.0f;
  for (size_t i = 0; i < n; i++) {
    float absErr = fabs(got[i] - ref[i]);
    float bound = atol + rtol * fabs(ref[i]);
    if (absErr > maxAbsErr) maxAbsErr = absErr;
    float relErr = absErr / (fabs(ref[i]) + 1e-6f);
    if (relErr > maxRelErr) maxRelErr = relErr;
    if (absErr > bound) {
      if (errors < 10) {
        printf("  MISMATCH [%zu]: got %f, expected %f (abs err %e)\n", i,
               got[i], ref[i], absErr);
      }
      errors++;
    }
  }
  printf("Max abs error: %e, max rel error: %e\n", maxAbsErr, maxRelErr);
  if (errors == 0) {
    printf("PASS: All %zu elements match (atol=%e, rtol=%e).\n", n, atol,
           rtol);
  } else {
    printf("FAIL: %d / %zu mismatched.\n", errors, n);
  }
  return errors;
}

struct Args {
  std::string hsacoPath;
  std::string kernelName;
  int gridX = 1, gridY = 1, gridZ = 1;
  int blockX = 256, blockY = 1, blockZ = 1;
  std::vector<std::string> inputs;
  std::string expectedOutput;  // @file.npy
  std::string output;          // @file.npy or "-"
  std::vector<uint32_t> pushConstants;
  float atol = 1e-5f;
  float rtol = 1e-5f;
  std::string wgCountLib;
  std::string wgCountFunc;
  std::string artifactsDir;  // --artifacts=dir
};

// Simple JSON value extractor for metadata.json.
// Finds "key": value and returns the value string (without quotes for strings).
static std::string jsonValue(const std::string &json, const char *key) {
  std::string needle = std::string("\"") + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return "";
  pos++;  // skip ':'
  while (pos < json.size() && json[pos] == ' ') pos++;
  if (pos >= json.size()) return "";
  if (json[pos] == '"') {
    // String value.
    auto end = json.find('"', pos + 1);
    return (end != std::string::npos) ? json.substr(pos + 1, end - pos - 1)
                                      : "";
  }
  if (json[pos] == '[') {
    // Array value (return including brackets).
    auto end = json.find(']', pos);
    return (end != std::string::npos) ? json.substr(pos, end - pos + 1) : "";
  }
  // Numeric value.
  auto end = pos;
  while (end < json.size() && json[end] != ',' && json[end] != '\n' &&
         json[end] != '}') {
    end++;
  }
  return json.substr(pos, end - pos);
}

// Load artifacts from --artifacts=dir: reads metadata.json and sets up paths.
static void loadArtifacts(Args &args) {
  std::string dir = args.artifactsDir;

  // Read metadata.json.
  std::string metaPath = dir + "/metadata.json";
  FILE *f = fopen(metaPath.c_str(), "r");
  if (!f) {
    fprintf(stderr, "Error: cannot open %s\n", metaPath.c_str());
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::string json(sz, '\0');
  fread(&json[0], 1, sz, f);
  fclose(f);

  // Set hsaco path.
  if (args.hsacoPath.empty()) {
    args.hsacoPath = dir + "/kernel.hsaco";
  }

  // Set kernel name from metadata.
  if (args.kernelName.empty()) {
    args.kernelName = jsonValue(json, "kernel_name");
  }

  // Set workgroup size from metadata.
  std::string wgStr = jsonValue(json, "workgroup_size");
  if (!wgStr.empty() && args.blockX == 256 && args.blockY == 1 &&
      args.blockZ == 1) {
    // Parse [x, y, z].
    sscanf(wgStr.c_str(), "[%d, %d, %d]", &args.blockX, &args.blockY,
           &args.blockZ);
  }

  // Set workgroup count .so path.
  std::string soPath = dir + "/workgroup_count.so";
  if (args.wgCountLib.empty()) {
    // Check if the .so exists before setting it.
    FILE *soFile = fopen(soPath.c_str(), "r");
    if (soFile) {
      fclose(soFile);
      args.wgCountLib = soPath;
    }
  }
}

static Args parseArgs(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto val = [&](const char *prefix) -> std::string {
      return a.substr(strlen(prefix));
    };
    if (a.rfind("--hsaco=", 0) == 0)
      args.hsacoPath = val("--hsaco=");
    else if (a.rfind("--kernel=", 0) == 0)
      args.kernelName = val("--kernel=");
    else if (a.rfind("--grid=", 0) == 0)
      sscanf(a.c_str() + 7, "%d,%d,%d", &args.gridX, &args.gridY, &args.gridZ);
    else if (a.rfind("--workgroup=", 0) == 0)
      sscanf(a.c_str() + 12, "%d,%d,%d", &args.blockX, &args.blockY,
             &args.blockZ);
    else if (a.rfind("--input=", 0) == 0)
      args.inputs.push_back(val("--input="));
    else if (a.rfind("--expected_output=", 0) == 0)
      args.expectedOutput = val("--expected_output=");
    else if (a.rfind("--output=", 0) == 0)
      args.output = val("--output=");
    else if (a.rfind("--push-constants=", 0) == 0) {
      std::string pcs = val("--push-constants=");
      char *p = (char *)pcs.c_str();
      while (*p) {
        args.pushConstants.push_back((uint32_t)strtoul(p, &p, 10));
        if (*p == ',') p++;
      }
    } else if (a.rfind("--atol=", 0) == 0)
      args.atol = atof(a.c_str() + 7);
    else if (a.rfind("--rtol=", 0) == 0)
      args.rtol = atof(a.c_str() + 7);
    else if (a.rfind("--wg-count-lib=", 0) == 0)
      args.wgCountLib = val("--wg-count-lib=");
    else if (a.rfind("--wg-count-func=", 0) == 0)
      args.wgCountFunc = val("--wg-count-func=");
    else if (a.rfind("--artifacts=", 0) == 0)
      args.artifactsDir = val("--artifacts=");
    else {
      fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      exit(1);
    }
  }

  // If --artifacts is specified, load metadata.json and set paths.
  if (!args.artifactsDir.empty()) {
    loadArtifacts(args);
  }

  if (args.hsacoPath.empty()) {
    fprintf(stderr, "Error: --hsaco=path or --artifacts=dir required\n");
    exit(1);
  }
  if (args.inputs.empty()) {
    fprintf(stderr, "Error: at least one --input required\n");
    exit(1);
  }
  return args;
}

int main(int argc, char **argv) {
  Args args = parseArgs(argc, argv);

  printf("=== run-hip-kernel ===\n");
  printf("HSACO:     %s\n", args.hsacoPath.c_str());
  printf("Kernel:    %s\n", args.kernelName.c_str());
  printf("Grid:      %d x %d x %d\n", args.gridX, args.gridY, args.gridZ);
  printf("Workgroup: %d x %d x %d\n", args.blockX, args.blockY, args.blockZ);
  printf("Inputs:    %zu\n", args.inputs.size());

  // Initialize HIP.
  HIP_CHECK(hipInit(0));
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "No HIP devices\n");
    return 1;
  }
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("Device:    %s (%s)\n", prop.name, prop.gcnArchName);

  // Load HSACO.
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, args.hsacoPath.c_str()));
  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, args.kernelName.c_str()));

  // Compute workgroup count from .so if provided.
  if (!args.wgCountLib.empty()) {
    void *lib = dlopen(args.wgCountLib.c_str(), RTLD_LAZY);
    if (!lib) {
      fprintf(stderr, "dlopen(%s): %s\n", args.wgCountLib.c_str(), dlerror());
      return 1;
    }
    std::string funcName = args.wgCountFunc.empty()
                               ? args.kernelName + "_workgroup_count"
                               : args.wgCountFunc;
    using WgCountFn = void (*)(int64_t *, const int64_t *);
    auto fn = (WgCountFn)dlsym(lib, funcName.c_str());
    if (!fn) {
      fprintf(stderr, "dlsym(%s): %s\n", funcName.c_str(), dlerror());
      dlclose(lib);
      return 1;
    }
    // Each push constant is one i32 loaded via hal.interface.constant.load
    // and cast to index. Pass them as i64 to the workgroup count function.
    std::vector<int64_t> wgArgs;
    for (auto pc : args.pushConstants) {
      wgArgs.push_back(static_cast<int64_t>(pc));
    }
    int64_t grid[3] = {1, 1, 1};
    fn(grid, wgArgs.data());
    args.gridX = static_cast<int>(grid[0]);
    args.gridY = static_cast<int>(grid[1]);
    args.gridZ = static_cast<int>(grid[2]);
    printf("Grid from %s: %d x %d x %d\n", funcName.c_str(), args.gridX,
           args.gridY, args.gridZ);
    dlclose(lib);
  }

  // Parse and allocate buffers.
  // The last input is the output buffer.
  srand(42);
  std::vector<Buffer> buffers;
  for (auto &desc : args.inputs) {
    buffers.push_back(parseBuffer(desc));
  }
  size_t outputIdx = buffers.size() - 1;

  for (auto &buf : buffers) {
    initBuffer(buf);
    HIP_CHECK(hipMalloc(&buf.devicePtr, buf.sizeBytes()));
    HIP_CHECK(hipMemcpy(buf.devicePtr, buf.hostData.data(), buf.sizeBytes(),
                        hipMemcpyHostToDevice));
  }

  // Build kernel args: buffer pointers then push constants.
  std::vector<uint8_t> argBuf;
  auto pushPtr = [&](void *p) {
    argBuf.insert(argBuf.end(), (uint8_t *)&p, (uint8_t *)&p + sizeof(void *));
  };
  auto pushU32 = [&](uint32_t v) {
    argBuf.insert(argBuf.end(), (uint8_t *)&v,
                  (uint8_t *)&v + sizeof(uint32_t));
  };
  for (auto &buf : buffers) {
    pushPtr(buf.devicePtr);
  }
  for (auto pc : args.pushConstants) {
    pushU32(pc);
  }

  size_t argSize = argBuf.size();
  void *argPtr = argBuf.data();
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argPtr,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};

  // Launch.
  printf("Launching kernel (%zu bytes of args)...\n", argSize);
  HIP_CHECK(hipModuleLaunchKernel(function, args.gridX, args.gridY, args.gridZ,
                                  args.blockX, args.blockY, args.blockZ, 0,
                                  nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  printf("Kernel completed.\n");

  // Read back output.
  Buffer &outBuf = buffers[outputIdx];
  std::vector<float> gpuResult(outBuf.numElements());
  HIP_CHECK(hipMemcpy(gpuResult.data(), outBuf.devicePtr, outBuf.sizeBytes(),
                      hipMemcpyDeviceToHost));

  int errors = 0;

  // Compare against expected output if provided.
  if (!args.expectedOutput.empty()) {
    std::vector<float> expected;
    std::string path = args.expectedOutput;
    if (path[0] == '@') path = path.substr(1);
    if (!loadNpy(path, expected)) {
      fprintf(stderr, "Error: cannot load expected output %s\n", path.c_str());
      return 1;
    }
    if (expected.size() != gpuResult.size()) {
      fprintf(stderr, "Error: expected %zu elements, got %zu\n",
              expected.size(), gpuResult.size());
      return 1;
    }
    errors = verify(gpuResult.data(), expected.data(), gpuResult.size(),
                    args.atol, args.rtol);
  }

  // Print output summary by default; save to file if requested.
  if (args.output.empty() && args.expectedOutput.empty()) {
    // No --output or --expected_output: print shape + summary.
    printf("Result: ");
    for (size_t i = 0; i < outBuf.shape.size(); i++) {
      if (i > 0) printf("x");
      printf("%ld", outBuf.shape[i]);
    }
    printf("x%s\n", outBuf.dtype.c_str());
    float minVal = gpuResult[0], maxVal = gpuResult[0];
    for (size_t i = 1; i < gpuResult.size(); i++) {
      if (gpuResult[i] < minVal) minVal = gpuResult[i];
      if (gpuResult[i] > maxVal) maxVal = gpuResult[i];
    }
    printf("  min=%f, max=%f, elements=%zu\n", minVal, maxVal,
           gpuResult.size());
  } else if (args.output == "-") {
    for (size_t i = 0; i < gpuResult.size() && i < 20; i++) {
      printf("  [%zu] = %f\n", i, gpuResult[i]);
    }
    if (gpuResult.size() > 20) {
      printf("  ... (%zu more elements)\n", gpuResult.size() - 20);
    }
  } else if (!args.output.empty()) {
    std::string path = args.output;
    if (path[0] == '@') path = path.substr(1);
    saveNpy(path, gpuResult.data(), gpuResult.size(), outBuf.shape);
    printf("Wrote output: %s\n", path.c_str());
  }

  // Cleanup.
  for (auto &buf : buffers) {
    HIP_CHECK(hipFree(buf.devicePtr));
  }
  HIP_CHECK(hipModuleUnload(module));
  return errors > 0 ? 1 : 0;
}
