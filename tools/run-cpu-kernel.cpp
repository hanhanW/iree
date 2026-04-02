// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Generic CPU kernel launcher. Zero IREE dependencies.
//
// Loads a shared library (.so), calls the kernel function with flat ABI
// (buffer pointers + push constants + workgroup args), and compares
// output against --expected_output.
//
// The kernel function has the signature:
//   void kernel(ptr b0, ..., i32 c0, ...,
//               i32 wg_id_x, i32 wg_id_y, i32 wg_id_z,
//               i32 wg_count_x, i32 wg_count_y, i32 wg_count_z)
//
// On x86-64 Linux, all arguments are passed as 64-bit values in registers
// and on the stack. We pack them into an int64_t array and use a generic
// trampoline to call the function.
//
// Usage:
//   run-cpu-kernel --artifacts=dir \
//       --input=4x128x256xf32=1 --input=4x256x512xf32=1 \
//       --input=4x128xf32=1 --input=4x128x512xf32 \
//       --expected_output=@expected.npy \
//       [--push-constants=val0,val1,...] [--threads=N]

#include <dlfcn.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// Minimum alignment for buffer allocations. Read from metadata.json
// (buffer_alignment field). Default 64 for AVX-512 safety.
static size_t gBufferAlignment = 64;

// A buffer described by shape + element type.
// Uses aligned memory allocation for SIMD compatibility.
struct Buffer {
  std::vector<int64_t> shape;
  std::string dtype;
  std::string source;
  float *hostData = nullptr;
  size_t hostDataSize = 0;

  ~Buffer() {
    if (hostData) {
      free(hostData);
    }
  }

  // Disable copy, allow move.
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept
      : shape(std::move(other.shape)), dtype(std::move(other.dtype)),
        source(std::move(other.source)), hostData(other.hostData),
        hostDataSize(other.hostDataSize) {
    other.hostData = nullptr;
    other.hostDataSize = 0;
  }
  Buffer() = default;

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

  void allocate() {
    hostDataSize = numElements();
    size_t sz = sizeBytes();
    // Round up to alignment boundary.
    sz = (sz + gBufferAlignment - 1) & ~(gBufferAlignment - 1);
    hostData = (float *)aligned_alloc(gBufferAlignment, sz);
    memset(hostData, 0, sz);
  }
};

static Buffer parseBuffer(const std::string &desc) {
  Buffer buf;
  std::string rest = desc;
  auto eq = rest.find('=');
  if (eq != std::string::npos) {
    buf.source = rest.substr(eq + 1);
    rest = rest.substr(0, eq);
  }
  size_t pos = 0;
  while (pos < rest.size()) {
    auto xpos = rest.find('x', pos);
    std::string tok = (xpos == std::string::npos)
                          ? rest.substr(pos)
                          : rest.substr(pos, xpos - pos);
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

static bool loadNpy(const std::string &path, std::vector<float> &data) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[6];
  if (fread(magic, 1, 6, f) != 6) { fclose(f); return false; }
  uint8_t major, minor;
  if (fread(&major, 1, 1, f) != 1) { fclose(f); return false; }
  if (fread(&minor, 1, 1, f) != 1) { fclose(f); return false; }
  uint32_t headerLen = 0;
  if (major >= 2) {
    if (fread(&headerLen, 4, 1, f) != 1) { fclose(f); return false; }
  } else {
    uint16_t hl16;
    if (fread(&hl16, 2, 1, f) != 1) { fclose(f); return false; }
    headerLen = hl16;
  }
  fseek(f, headerLen, SEEK_CUR);
  long dataStart = ftell(f);
  fseek(f, 0, SEEK_END);
  long dataSize = ftell(f) - dataStart;
  fseek(f, dataStart, SEEK_SET);
  data.resize(dataSize / sizeof(float));
  size_t read = fread(data.data(), sizeof(float), data.size(), f);
  fclose(f);
  return read == data.size();
}

static void initBuffer(Buffer &buf) {
  buf.allocate();
  size_t n = buf.numElements();
  if (buf.source.empty()) {
    srand(42);
    for (size_t i = 0; i < n; i++) {
      buf.hostData[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }
  } else if (buf.source == "0") {
    // Already zeroed by allocate().
  } else if (buf.source == "1") {
    for (size_t i = 0; i < n; i++) {
      buf.hostData[i] = 1.0f;
    }
  } else if (buf.source[0] == '@') {
    std::vector<float> tmpData;
    if (!loadNpy(buf.source.substr(1), tmpData)) {
      fprintf(stderr, "Error: cannot load %s\n", buf.source.c_str() + 1);
      exit(1);
    }
    memcpy(buf.hostData, tmpData.data(), tmpData.size() * sizeof(float));
  } else {
    float val = std::stof(buf.source);
    for (size_t i = 0; i < n; i++) {
      buf.hostData[i] = val;
    }
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

static std::string jsonValue(const std::string &json, const char *key) {
  std::string needle = std::string("\"") + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return "";
  pos++;
  while (pos < json.size() && json[pos] == ' ') pos++;
  if (pos >= json.size()) return "";
  if (json[pos] == '"') {
    auto end = json.find('"', pos + 1);
    return (end != std::string::npos) ? json.substr(pos + 1, end - pos - 1)
                                      : "";
  }
  if (json[pos] == '[') {
    auto end = json.find(']', pos);
    return (end != std::string::npos) ? json.substr(pos, end - pos + 1) : "";
  }
  auto end = pos;
  while (end < json.size() && json[end] != ',' && json[end] != '\n' &&
         json[end] != '}') {
    end++;
  }
  return json.substr(pos, end - pos);
}

struct Args {
  std::string soPath;
  std::string kernelName;
  int gridX = 1, gridY = 1, gridZ = 1;
  std::vector<std::string> inputs;
  std::string expectedOutput;
  std::string output;
  std::vector<uint32_t> pushConstants;
  float atol = 1e-5f;
  float rtol = 1e-5f;
  std::string wgCountLib;
  std::string wgCountFunc;
  std::string artifactsDir;
  int numBindings = 0;
  int numConstants = 0;
  int bufferAlignment = 64;
  int threads = 0;  // 0 = auto
};

static void loadArtifacts(Args &args) {
  std::string dir = args.artifactsDir;
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

  std::string kernelFile = jsonValue(json, "kernel_file");
  if (args.soPath.empty()) {
    args.soPath = dir + "/" + (kernelFile.empty() ? "kernel.so" : kernelFile);
  }
  if (args.kernelName.empty()) {
    args.kernelName = jsonValue(json, "kernel_name");
  }
  std::string nbStr = jsonValue(json, "num_bindings");
  if (!nbStr.empty()) args.numBindings = std::stoi(nbStr);
  std::string ncStr = jsonValue(json, "num_constants");
  if (!ncStr.empty()) args.numConstants = std::stoi(ncStr);
  std::string baStr = jsonValue(json, "buffer_alignment");
  if (!baStr.empty()) args.bufferAlignment = std::stoi(baStr);

  std::string soPath = dir + "/workgroup_count.so";
  if (args.wgCountLib.empty()) {
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
    if (a.rfind("--so=", 0) == 0)
      args.soPath = val("--so=");
    else if (a.rfind("--kernel=", 0) == 0)
      args.kernelName = val("--kernel=");
    else if (a.rfind("--grid=", 0) == 0)
      sscanf(a.c_str() + 7, "%d,%d,%d", &args.gridX, &args.gridY, &args.gridZ);
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
    else if (a.rfind("--threads=", 0) == 0)
      args.threads = std::stoi(val("--threads="));
    else {
      fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      exit(1);
    }
  }

  if (!args.artifactsDir.empty()) {
    loadArtifacts(args);
  }
  if (args.soPath.empty()) {
    fprintf(stderr, "Error: --so=path or --artifacts=dir required\n");
    exit(1);
  }
  if (args.inputs.empty()) {
    fprintf(stderr, "Error: at least one --input required\n");
    exit(1);
  }
  return args;
}

// Call the kernel function with packed arguments.
// The flat ABI kernel signature on x86-64 is:
//   void kernel(void* b0, ..., uint32_t c0, ...,
//               uint32_t wg_id_x, wg_id_y, wg_id_z,
//               uint32_t wg_count_x, wg_count_y, wg_count_z)
//
// On x86-64, all integer/pointer args <=8 bytes are passed in registers
// (rdi, rsi, rdx, rcx, r8, r9) then on the stack, zero-extended to 64 bits.
// We pack all args as int64_t and use a function pointer cast.
static void callKernel(void *fnPtr, void **bindingPtrs, int numBindings,
                       uint32_t *constants, int numConstants,
                       uint32_t wgIdX, uint32_t wgIdY, uint32_t wgIdZ,
                       uint32_t wgCountX, uint32_t wgCountY,
                       uint32_t wgCountZ) {
  // Pack all arguments as int64_t (zero-extended for uint32_t).
  std::vector<int64_t> args;
  for (int i = 0; i < numBindings; ++i) {
    args.push_back(reinterpret_cast<int64_t>(bindingPtrs[i]));
  }
  for (int i = 0; i < numConstants; ++i) {
    args.push_back(static_cast<int64_t>(constants[i]));
  }
  args.push_back(static_cast<int64_t>(wgIdX));
  args.push_back(static_cast<int64_t>(wgIdY));
  args.push_back(static_cast<int64_t>(wgIdZ));
  args.push_back(static_cast<int64_t>(wgCountX));
  args.push_back(static_cast<int64_t>(wgCountY));
  args.push_back(static_cast<int64_t>(wgCountZ));

  // Call with the right number of arguments using a generated switch.
  // Supports up to 32 args (bindings + constants + 6 workgroup args).
  int n = args.size();
  auto a = args.data();

#define CALL(N, ...) \
  case N: ((void (*)(__VA_ARGS__))fnPtr)(__VA_ARGS__); break;
#define A(i) a[i]
#define T int64_t
  // clang-format off
  switch (n) {
  case 6:  { using F = void(*)(T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5)); break; }
  case 7:  { using F = void(*)(T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6)); break; }
  case 8:  { using F = void(*)(T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7)); break; }
  case 9:  { using F = void(*)(T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8)); break; }
  case 10: { using F = void(*)(T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9)); break; }
  case 11: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10)); break; }
  case 12: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11)); break; }
  case 13: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12)); break; }
  case 14: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12),A(13)); break; }
  case 15: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12),A(13),A(14)); break; }
  case 16: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12),A(13),A(14),A(15)); break; }
  case 17: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12),A(13),A(14),A(15),A(16)); break; }
  case 18: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12),A(13),A(14),A(15),A(16),A(17)); break; }
  case 19: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12),A(13),A(14),A(15),A(16),A(17),A(18)); break; }
  case 20: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12),A(13),A(14),A(15),A(16),A(17),A(18),A(19)); break; }
  case 21: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12),A(13),A(14),A(15),A(16),A(17),A(18),A(19),A(20)); break; }
  case 22: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12),A(13),A(14),A(15),A(16),A(17),A(18),A(19),A(20),A(21)); break; }
  case 23: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12),A(13),A(14),A(15),A(16),A(17),A(18),A(19),A(20),A(21),A(22)); break; }
  case 24: { using F = void(*)(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T); ((F)fnPtr)(A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),A(11),A(12),A(13),A(14),A(15),A(16),A(17),A(18),A(19),A(20),A(21),A(22),A(23)); break; }
  default:
    fprintf(stderr, "Error: unsupported arg count %d (max 24)\n", n);
    exit(1);
  }
  // clang-format on
#undef A
#undef T
}

int main(int argc, char **argv) {
  Args args = parseArgs(argc, argv);
  gBufferAlignment = args.bufferAlignment;

  printf("=== run-cpu-kernel ===\n");
  printf("Library:   %s\n", args.soPath.c_str());
  printf("Kernel:    %s\n", args.kernelName.c_str());
  printf("Bindings:  %d\n", args.numBindings);
  printf("Constants: %d\n", args.numConstants);
  printf("Inputs:    %zu\n", args.inputs.size());

  // Load shared library.
  void *lib = dlopen(args.soPath.c_str(), RTLD_LAZY);
  if (!lib) {
    fprintf(stderr, "dlopen(%s): %s\n", args.soPath.c_str(), dlerror());
    return 1;
  }
  void *fnPtr = dlsym(lib, args.kernelName.c_str());
  if (!fnPtr) {
    fprintf(stderr, "dlsym(%s): %s\n", args.kernelName.c_str(), dlerror());
    dlclose(lib);
    return 1;
  }
  printf("Loaded kernel: %s @ %p\n", args.kernelName.c_str(), fnPtr);

  // Compute workgroup count from .so if provided.
  if (!args.wgCountLib.empty()) {
    void *wgLib = dlopen(args.wgCountLib.c_str(), RTLD_LAZY);
    if (!wgLib) {
      fprintf(stderr, "dlopen(%s): %s\n", args.wgCountLib.c_str(), dlerror());
      return 1;
    }
    std::string funcName = args.wgCountFunc.empty()
                               ? args.kernelName + "_workgroup_count"
                               : args.wgCountFunc;
    using WgCountFn = void (*)(int64_t *, const int64_t *);
    auto fn = (WgCountFn)dlsym(wgLib, funcName.c_str());
    if (!fn) {
      fprintf(stderr, "dlsym(%s): %s\n", funcName.c_str(), dlerror());
      dlclose(wgLib);
      return 1;
    }
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
    dlclose(wgLib);
  }

  printf("Grid:      %d x %d x %d\n", args.gridX, args.gridY, args.gridZ);

  // Parse and allocate buffers.
  srand(42);
  std::vector<Buffer> buffers;
  for (auto &desc : args.inputs) {
    buffers.push_back(parseBuffer(desc));
  }
  size_t outputIdx = buffers.size() - 1;
  for (auto &buf : buffers) {
    initBuffer(buf);
  }

  // Build binding pointer array.
  std::vector<void *> bindingPtrs;
  for (auto &buf : buffers) {
    bindingPtrs.push_back(buf.hostData);
  }

  // Dispatch workgroups across threads.
  int totalWGs = args.gridX * args.gridY * args.gridZ;
  int numThreads = args.threads > 0
                       ? args.threads
                       : std::min(totalWGs,
                                  (int)std::thread::hardware_concurrency());
  if (numThreads < 1) numThreads = 1;

  printf("Dispatching %d workgroups across %d threads...\n", totalWGs,
         numThreads);

  std::vector<std::thread> threads;
  for (int t = 0; t < numThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int wg = t; wg < totalWGs; wg += numThreads) {
        int wgZ = wg / (args.gridX * args.gridY);
        int wgY = (wg / args.gridX) % args.gridY;
        int wgX = wg % args.gridX;
        callKernel(fnPtr, bindingPtrs.data(), args.numBindings,
                   args.pushConstants.data(), args.numConstants,
                   wgX, wgY, wgZ, args.gridX, args.gridY, args.gridZ);
      }
    });
  }
  for (auto &t : threads) {
    t.join();
  }
  printf("Kernel completed.\n");

  // Get output buffer.
  Buffer &outBuf = buffers[outputIdx];
  const float *result = outBuf.hostData;

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
    if (expected.size() != outBuf.numElements()) {
      fprintf(stderr, "Error: expected %zu elements, got %zu\n",
              expected.size(), outBuf.numElements());
      return 1;
    }
    errors = verify(result, expected.data(), outBuf.numElements(), args.atol,
                    args.rtol);
  }

  // Print output summary.
  if (args.output.empty() && args.expectedOutput.empty()) {
    printf("Result: ");
    for (size_t i = 0; i < outBuf.shape.size(); i++) {
      if (i > 0) printf("x");
      printf("%ld", outBuf.shape[i]);
    }
    printf("x%s\n", outBuf.dtype.c_str());
    float minVal = result[0], maxVal = result[0];
    for (size_t i = 1; i < outBuf.numElements(); i++) {
      if (result[i] < minVal) minVal = result[i];
      if (result[i] > maxVal) maxVal = result[i];
    }
    printf("  min=%f, max=%f, elements=%zu\n", minVal, maxVal,
           outBuf.numElements());
  } else if (args.output == "-") {
    for (size_t i = 0; i < outBuf.numElements() && i < 20; i++) {
      printf("  [%zu] = %f\n", i, result[i]);
    }
    if (outBuf.numElements() > 20) {
      printf("  ... (%zu more elements)\n", outBuf.numElements() - 20);
    }
  }

  dlclose(lib);
  return errors > 0 ? 1 : 0;
}
