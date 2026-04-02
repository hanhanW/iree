// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Standalone Vulkan compute kernel launcher. Zero IREE dependencies.
//
// Loads a raw SPIR-V binary (.spv), creates Vulkan pipeline with descriptor
// sets for storage buffers and push constants, dispatches workgroups, and
// verifies output against --expected_output.
//
// Vulkan functions are loaded dynamically via dlopen to avoid build-time
// Vulkan SDK dependency.
//
// Usage:
//   run-vulkan-kernel --artifacts=dir \
//       --input=4x128x256xf32=1 --input=4x256x512xf32=1 \
//       --input=4x128xf32=1 --input=4x128x512xf32 \
//       --expected_output=@expected.npy \
//       [--push-constants=val0,val1,...] [--device=0]

#define VK_NO_PROTOTYPES
#include "third_party/vulkan_headers/include/vulkan/vulkan_core.h"

#include <dlfcn.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

//===----------------------------------------------------------------------===//
// Vulkan function loader
//===----------------------------------------------------------------------===//

static void *vkLib = nullptr;

#define VK_FUNC(name) static PFN_##name name = nullptr;
VK_FUNC(vkGetInstanceProcAddr)
VK_FUNC(vkCreateInstance)
VK_FUNC(vkDestroyInstance)
VK_FUNC(vkEnumeratePhysicalDevices)
VK_FUNC(vkGetPhysicalDeviceProperties)
VK_FUNC(vkGetPhysicalDeviceMemoryProperties)
VK_FUNC(vkCreateDevice)
VK_FUNC(vkDestroyDevice)
VK_FUNC(vkGetDeviceQueue)
VK_FUNC(vkCreateShaderModule)
VK_FUNC(vkDestroyShaderModule)
VK_FUNC(vkCreateDescriptorSetLayout)
VK_FUNC(vkDestroyDescriptorSetLayout)
VK_FUNC(vkCreatePipelineLayout)
VK_FUNC(vkDestroyPipelineLayout)
VK_FUNC(vkCreateComputePipelines)
VK_FUNC(vkDestroyPipeline)
VK_FUNC(vkCreateDescriptorPool)
VK_FUNC(vkDestroyDescriptorPool)
VK_FUNC(vkAllocateDescriptorSets)
VK_FUNC(vkUpdateDescriptorSets)
VK_FUNC(vkCreateBuffer)
VK_FUNC(vkDestroyBuffer)
VK_FUNC(vkGetBufferMemoryRequirements)
VK_FUNC(vkAllocateMemory)
VK_FUNC(vkFreeMemory)
VK_FUNC(vkBindBufferMemory)
VK_FUNC(vkMapMemory)
VK_FUNC(vkUnmapMemory)
VK_FUNC(vkCreateCommandPool)
VK_FUNC(vkDestroyCommandPool)
VK_FUNC(vkAllocateCommandBuffers)
VK_FUNC(vkBeginCommandBuffer)
VK_FUNC(vkEndCommandBuffer)
VK_FUNC(vkCmdBindPipeline)
VK_FUNC(vkCmdBindDescriptorSets)
VK_FUNC(vkCmdPushConstants)
VK_FUNC(vkCmdDispatch)
VK_FUNC(vkQueueSubmit)
VK_FUNC(vkQueueWaitIdle)
VK_FUNC(vkCreateFence)
VK_FUNC(vkDestroyFence)
VK_FUNC(vkWaitForFences)
#undef VK_FUNC

static bool loadVulkan() {
  vkLib = dlopen("libvulkan.so.1", RTLD_LAZY);
  if (!vkLib) vkLib = dlopen("libvulkan.so", RTLD_LAZY);
  if (!vkLib) {
    fprintf(stderr, "Error: cannot load libvulkan.so: %s\n", dlerror());
    return false;
  }
  vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(vkLib, "vkGetInstanceProcAddr");
  if (!vkGetInstanceProcAddr) {
    fprintf(stderr, "Error: cannot find vkGetInstanceProcAddr\n");
    return false;
  }
#define LOAD(name)                                                             \
  name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);             \
  if (!name) {                                                                 \
    fprintf(stderr, "Warning: cannot load %s (pre-instance)\n", #name);        \
  }
  LOAD(vkCreateInstance)
#undef LOAD
  return true;
}

static void loadInstanceFuncs(VkInstance instance) {
#define LOAD(name)                                                             \
  name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);
  LOAD(vkDestroyInstance)
  LOAD(vkEnumeratePhysicalDevices)
  LOAD(vkGetPhysicalDeviceProperties)
  LOAD(vkGetPhysicalDeviceMemoryProperties)
  LOAD(vkCreateDevice)
  LOAD(vkDestroyDevice)
  LOAD(vkGetDeviceQueue)
  LOAD(vkCreateShaderModule)
  LOAD(vkDestroyShaderModule)
  LOAD(vkCreateDescriptorSetLayout)
  LOAD(vkDestroyDescriptorSetLayout)
  LOAD(vkCreatePipelineLayout)
  LOAD(vkDestroyPipelineLayout)
  LOAD(vkCreateComputePipelines)
  LOAD(vkDestroyPipeline)
  LOAD(vkCreateDescriptorPool)
  LOAD(vkDestroyDescriptorPool)
  LOAD(vkAllocateDescriptorSets)
  LOAD(vkUpdateDescriptorSets)
  LOAD(vkCreateBuffer)
  LOAD(vkDestroyBuffer)
  LOAD(vkGetBufferMemoryRequirements)
  LOAD(vkAllocateMemory)
  LOAD(vkFreeMemory)
  LOAD(vkBindBufferMemory)
  LOAD(vkMapMemory)
  LOAD(vkUnmapMemory)
  LOAD(vkCreateCommandPool)
  LOAD(vkDestroyCommandPool)
  LOAD(vkAllocateCommandBuffers)
  LOAD(vkBeginCommandBuffer)
  LOAD(vkEndCommandBuffer)
  LOAD(vkCmdBindPipeline)
  LOAD(vkCmdBindDescriptorSets)
  LOAD(vkCmdPushConstants)
  LOAD(vkCmdDispatch)
  LOAD(vkQueueSubmit)
  LOAD(vkQueueWaitIdle)
  LOAD(vkCreateFence)
  LOAD(vkDestroyFence)
  LOAD(vkWaitForFences)
#undef LOAD
}

#define VK_CHECK(expr)                                                         \
  do {                                                                         \
    VkResult _r = (expr);                                                      \
    if (_r != VK_SUCCESS) {                                                    \
      fprintf(stderr, "Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__);  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

//===----------------------------------------------------------------------===//
// Buffer / numpy helpers (shared with run-hip-kernel / run-cpu-kernel)
//===----------------------------------------------------------------------===//

struct Buffer {
  std::vector<int64_t> shape;
  std::string dtype;
  std::string source;
  std::vector<float> hostData;

  size_t numElements() const {
    size_t n = 1;
    for (auto d : shape) n *= d;
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
    std::string tok =
        (xpos == std::string::npos) ? rest.substr(pos) : rest.substr(pos, xpos - pos);
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
  size_t n = buf.numElements();
  buf.hostData.resize(n);
  if (buf.source.empty()) {
    srand(42);
    for (size_t i = 0; i < n; i++)
      buf.hostData[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
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
    float val = std::stof(buf.source);
    std::fill(buf.hostData.begin(), buf.hostData.end(), val);
  }
}

static int verify(const float *got, const float *ref, size_t n, float atol,
                  float rtol) {
  int errors = 0;
  float maxAbsErr = 0, maxRelErr = 0;
  for (size_t i = 0; i < n; i++) {
    float absErr = fabs(got[i] - ref[i]);
    float bound = atol + rtol * fabs(ref[i]);
    if (absErr > maxAbsErr) maxAbsErr = absErr;
    float relErr = absErr / (fabs(ref[i]) + 1e-6f);
    if (relErr > maxRelErr) maxRelErr = relErr;
    if (absErr > bound) {
      if (errors < 10)
        printf("  MISMATCH [%zu]: got %f, expected %f (abs err %e)\n", i,
               got[i], ref[i], absErr);
      errors++;
    }
  }
  printf("Max abs error: %e, max rel error: %e\n", maxAbsErr, maxRelErr);
  if (errors == 0)
    printf("PASS: All %zu elements match (atol=%e, rtol=%e).\n", n, atol, rtol);
  else
    printf("FAIL: %d / %zu mismatched.\n", errors, n);
  return errors;
}

//===----------------------------------------------------------------------===//
// JSON + args (shared pattern)
//===----------------------------------------------------------------------===//

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
    return (end != std::string::npos) ? json.substr(pos + 1, end - pos - 1) : "";
  }
  if (json[pos] == '[') {
    auto end = json.find(']', pos);
    return (end != std::string::npos) ? json.substr(pos, end - pos + 1) : "";
  }
  auto end = pos;
  while (end < json.size() && json[end] != ',' && json[end] != '\n' &&
         json[end] != '}')
    end++;
  return json.substr(pos, end - pos);
}

struct Args {
  std::string spvPath;
  std::string kernelName;
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
  int blockX = 1, blockY = 1, blockZ = 1;
  int gridX = 1, gridY = 1, gridZ = 1;
  int deviceIndex = 0;
};

static void loadArtifacts(Args &args) {
  std::string dir = args.artifactsDir;
  std::string metaPath = dir + "/metadata.json";
  FILE *f = fopen(metaPath.c_str(), "r");
  if (!f) { fprintf(stderr, "Error: cannot open %s\n", metaPath.c_str()); exit(1); }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::string json(sz, '\0');
  fread(&json[0], 1, sz, f);
  fclose(f);

  std::string kernelFile = jsonValue(json, "kernel_file");
  if (args.spvPath.empty())
    args.spvPath = dir + "/" + (kernelFile.empty() ? "kernel.spv" : kernelFile);
  if (args.kernelName.empty())
    args.kernelName = jsonValue(json, "kernel_name");
  std::string nbStr = jsonValue(json, "num_bindings");
  if (!nbStr.empty()) args.numBindings = std::stoi(nbStr);
  std::string ncStr = jsonValue(json, "num_constants");
  if (!ncStr.empty()) args.numConstants = std::stoi(ncStr);
  std::string wgStr = jsonValue(json, "workgroup_size");
  if (!wgStr.empty())
    sscanf(wgStr.c_str(), "[%d, %d, %d]", &args.blockX, &args.blockY, &args.blockZ);

  std::string soPath = dir + "/workgroup_count.so";
  if (args.wgCountLib.empty()) {
    FILE *soFile = fopen(soPath.c_str(), "r");
    if (soFile) { fclose(soFile); args.wgCountLib = soPath; }
  }
}

static Args parseArgs(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto val = [&](const char *pfx) -> std::string { return a.substr(strlen(pfx)); };
    if (a.rfind("--spv=", 0) == 0) args.spvPath = val("--spv=");
    else if (a.rfind("--kernel=", 0) == 0) args.kernelName = val("--kernel=");
    else if (a.rfind("--input=", 0) == 0) args.inputs.push_back(val("--input="));
    else if (a.rfind("--expected_output=", 0) == 0) args.expectedOutput = val("--expected_output=");
    else if (a.rfind("--output=", 0) == 0) args.output = val("--output=");
    else if (a.rfind("--push-constants=", 0) == 0) {
      std::string pcs = val("--push-constants=");
      char *p = (char *)pcs.c_str();
      while (*p) { args.pushConstants.push_back((uint32_t)strtoul(p, &p, 10)); if (*p == ',') p++; }
    }
    else if (a.rfind("--atol=", 0) == 0) args.atol = atof(a.c_str() + 7);
    else if (a.rfind("--rtol=", 0) == 0) args.rtol = atof(a.c_str() + 7);
    else if (a.rfind("--wg-count-lib=", 0) == 0) args.wgCountLib = val("--wg-count-lib=");
    else if (a.rfind("--wg-count-func=", 0) == 0) args.wgCountFunc = val("--wg-count-func=");
    else if (a.rfind("--artifacts=", 0) == 0) args.artifactsDir = val("--artifacts=");
    else if (a.rfind("--device=", 0) == 0) args.deviceIndex = std::stoi(val("--device="));
    else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); exit(1); }
  }
  if (!args.artifactsDir.empty()) loadArtifacts(args);
  if (args.spvPath.empty()) { fprintf(stderr, "Error: --spv=path or --artifacts=dir required\n"); exit(1); }
  if (args.inputs.empty()) { fprintf(stderr, "Error: at least one --input required\n"); exit(1); }
  return args;
}

//===----------------------------------------------------------------------===//
// Vulkan helpers
//===----------------------------------------------------------------------===//

static uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t typeBits,
                                VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if ((typeBits & (1 << i)) &&
        (memProps.memoryTypes[i].propertyFlags & props) == props)
      return i;
  }
  fprintf(stderr, "Error: no suitable memory type\n");
  exit(1);
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  Args args = parseArgs(argc, argv);

  printf("=== run-vulkan-kernel ===\n");
  printf("SPIR-V:    %s\n", args.spvPath.c_str());
  printf("Kernel:    %s\n", args.kernelName.c_str());
  printf("Bindings:  %d\n", args.numBindings);
  printf("Constants: %d\n", args.numConstants);
  printf("Block:     %d x %d x %d\n", args.blockX, args.blockY, args.blockZ);

  // Load SPIR-V binary.
  FILE *spvFile = fopen(args.spvPath.c_str(), "rb");
  if (!spvFile) { fprintf(stderr, "Error: cannot open %s\n", args.spvPath.c_str()); return 1; }
  fseek(spvFile, 0, SEEK_END);
  long spvSize = ftell(spvFile);
  fseek(spvFile, 0, SEEK_SET);
  std::vector<uint32_t> spvCode(spvSize / sizeof(uint32_t));
  fread(spvCode.data(), sizeof(uint32_t), spvCode.size(), spvFile);
  fclose(spvFile);
  printf("SPIR-V:    %zu words\n", spvCode.size());

  // Load Vulkan.
  if (!loadVulkan()) return 1;

  // Create instance.
  VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  appInfo.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo instInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instInfo.pApplicationInfo = &appInfo;
  VkInstance instance;
  VK_CHECK(vkCreateInstance(&instInfo, nullptr, &instance));
  loadInstanceFuncs(instance);

  // Select physical device.
  uint32_t physDevCount = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &physDevCount, nullptr));
  if (physDevCount == 0) { fprintf(stderr, "No Vulkan devices\n"); return 1; }
  std::vector<VkPhysicalDevice> physDevs(physDevCount);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &physDevCount, physDevs.data()));
  VkPhysicalDevice physDev = physDevs[args.deviceIndex];
  VkPhysicalDeviceProperties devProps;
  vkGetPhysicalDeviceProperties(physDev, &devProps);
  printf("Device:    %s\n", devProps.deviceName);

  // Create logical device with a compute queue.
  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queueInfo.queueFamilyIndex = 0; // assume family 0 supports compute
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &queuePriority;
  VkDeviceCreateInfo devInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  devInfo.queueCreateInfoCount = 1;
  devInfo.pQueueCreateInfos = &queueInfo;
  VkDevice device;
  VK_CHECK(vkCreateDevice(physDev, &devInfo, nullptr, &device));
  VkQueue queue;
  vkGetDeviceQueue(device, 0, 0, &queue);

  // Compute workgroup count.
  if (!args.wgCountLib.empty()) {
    void *wgLib = dlopen(args.wgCountLib.c_str(), RTLD_LAZY);
    if (!wgLib) { fprintf(stderr, "dlopen(%s): %s\n", args.wgCountLib.c_str(), dlerror()); return 1; }
    std::string funcName = args.wgCountFunc.empty()
                               ? args.kernelName + "_workgroup_count"
                               : args.wgCountFunc;
    using WgCountFn = void (*)(int64_t *, const int64_t *);
    auto fn = (WgCountFn)dlsym(wgLib, funcName.c_str());
    if (!fn) { fprintf(stderr, "dlsym(%s): %s\n", funcName.c_str(), dlerror()); return 1; }
    std::vector<int64_t> wgArgs;
    for (auto pc : args.pushConstants) wgArgs.push_back(static_cast<int64_t>(pc));
    int64_t grid[3] = {1, 1, 1};
    fn(grid, wgArgs.data());
    args.gridX = static_cast<int>(grid[0]);
    args.gridY = static_cast<int>(grid[1]);
    args.gridZ = static_cast<int>(grid[2]);
    printf("Grid from %s: %d x %d x %d\n", funcName.c_str(), args.gridX, args.gridY, args.gridZ);
    dlclose(wgLib);
  }
  printf("Grid:      %d x %d x %d\n", args.gridX, args.gridY, args.gridZ);

  // Parse and initialize buffers.
  srand(42);
  std::vector<Buffer> buffers;
  for (auto &desc : args.inputs) buffers.push_back(parseBuffer(desc));
  size_t outputIdx = buffers.size() - 1;
  for (auto &buf : buffers) initBuffer(buf);

  // Create Vulkan buffers.
  struct VkBuf {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
  };
  std::vector<VkBuf> vkBufs(buffers.size());
  for (size_t i = 0; i < buffers.size(); i++) {
    VkBufferCreateInfo bufCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufCI.size = buffers[i].sizeBytes();
    bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bufCI, nullptr, &vkBufs[i].buffer));
    vkBufs[i].size = bufCI.size;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, vkBufs[i].buffer, &memReq);
    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(
        physDev, memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &vkBufs[i].memory));
    VK_CHECK(vkBindBufferMemory(device, vkBufs[i].buffer, vkBufs[i].memory, 0));

    // Upload data.
    void *mapped;
    VK_CHECK(vkMapMemory(device, vkBufs[i].memory, 0, bufCI.size, 0, &mapped));
    memcpy(mapped, buffers[i].hostData.data(), bufCI.size);
    vkUnmapMemory(device, vkBufs[i].memory);
  }

  // Create shader module.
  VkShaderModuleCreateInfo shaderCI = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  shaderCI.codeSize = spvCode.size() * sizeof(uint32_t);
  shaderCI.pCode = spvCode.data();
  VkShaderModule shaderModule;
  VK_CHECK(vkCreateShaderModule(device, &shaderCI, nullptr, &shaderModule));

  // Create descriptor set layout (all bindings are storage buffers).
  std::vector<VkDescriptorSetLayoutBinding> dsBindings(args.numBindings);
  for (int i = 0; i < args.numBindings; i++) {
    dsBindings[i] = {};
    dsBindings[i].binding = i;
    dsBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsBindings[i].descriptorCount = 1;
    dsBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dsLayoutCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dsLayoutCI.bindingCount = dsBindings.size();
  dsLayoutCI.pBindings = dsBindings.data();
  VkDescriptorSetLayout dsLayout;
  VK_CHECK(vkCreateDescriptorSetLayout(device, &dsLayoutCI, nullptr, &dsLayout));

  // Create pipeline layout (descriptor set + push constants).
  VkPushConstantRange pcRange = {};
  pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcRange.offset = 0;
  pcRange.size = args.numConstants * sizeof(uint32_t);
  VkPipelineLayoutCreateInfo plCI = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plCI.setLayoutCount = 1;
  plCI.pSetLayouts = &dsLayout;
  if (args.numConstants > 0) {
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges = &pcRange;
  }
  VkPipelineLayout pipelineLayout;
  VK_CHECK(vkCreatePipelineLayout(device, &plCI, nullptr, &pipelineLayout));

  // Create compute pipeline.
  VkComputePipelineCreateInfo cpCI = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpCI.stage.module = shaderModule;
  cpCI.stage.pName = args.kernelName.c_str();
  cpCI.layout = pipelineLayout;
  VkPipeline pipeline;
  VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpCI, nullptr, &pipeline));

  // Create descriptor pool and set.
  VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    static_cast<uint32_t>(args.numBindings)};
  VkDescriptorPoolCreateInfo dpCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpCI.maxSets = 1;
  dpCI.poolSizeCount = 1;
  dpCI.pPoolSizes = &poolSize;
  VkDescriptorPool descPool;
  VK_CHECK(vkCreateDescriptorPool(device, &dpCI, nullptr, &descPool));

  VkDescriptorSetAllocateInfo dsAllocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsAllocInfo.descriptorPool = descPool;
  dsAllocInfo.descriptorSetCount = 1;
  dsAllocInfo.pSetLayouts = &dsLayout;
  VkDescriptorSet descSet;
  VK_CHECK(vkAllocateDescriptorSets(device, &dsAllocInfo, &descSet));

  // Update descriptor set with buffer bindings.
  std::vector<VkDescriptorBufferInfo> bufInfos(args.numBindings);
  std::vector<VkWriteDescriptorSet> writes(args.numBindings);
  for (int i = 0; i < args.numBindings; i++) {
    bufInfos[i] = {vkBufs[i].buffer, 0, VK_WHOLE_SIZE};
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[i].dstSet = descSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufInfos[i];
  }
  vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);

  // Create command pool and buffer.
  VkCommandPoolCreateInfo cmdPoolCI = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cmdPoolCI.queueFamilyIndex = 0;
  VkCommandPool cmdPool;
  VK_CHECK(vkCreateCommandPool(device, &cmdPoolCI, nullptr, &cmdPool));

  VkCommandBufferAllocateInfo cmdBufAI = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmdBufAI.commandPool = cmdPool;
  cmdBufAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdBufAI.commandBufferCount = 1;
  VkCommandBuffer cmdBuf;
  VK_CHECK(vkAllocateCommandBuffers(device, &cmdBufAI, &cmdBuf));

  // Record command buffer.
  VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelineLayout, 0, 1, &descSet, 0, nullptr);
  if (args.numConstants > 0) {
    vkCmdPushConstants(cmdBuf, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       args.pushConstants.size() * sizeof(uint32_t),
                       args.pushConstants.data());
  }
  printf("Dispatching %d x %d x %d workgroups...\n", args.gridX, args.gridY, args.gridZ);
  vkCmdDispatch(cmdBuf, args.gridX, args.gridY, args.gridZ);
  VK_CHECK(vkEndCommandBuffer(cmdBuf));

  // Submit and wait.
  VkFenceCreateInfo fenceCI = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  VK_CHECK(vkCreateFence(device, &fenceCI, nullptr, &fence));
  VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuf;
  VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
  VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
  printf("Kernel completed.\n");

  // Read back output buffer.
  Buffer &outBuf = buffers[outputIdx];
  std::vector<float> result(outBuf.numElements());
  {
    void *mapped;
    VK_CHECK(vkMapMemory(device, vkBufs[outputIdx].memory, 0,
                          outBuf.sizeBytes(), 0, &mapped));
    memcpy(result.data(), mapped, outBuf.sizeBytes());
    vkUnmapMemory(device, vkBufs[outputIdx].memory);
  }

  int errors = 0;
  if (!args.expectedOutput.empty()) {
    std::vector<float> expected;
    std::string path = args.expectedOutput;
    if (path[0] == '@') path = path.substr(1);
    if (!loadNpy(path, expected)) {
      fprintf(stderr, "Error: cannot load %s\n", path.c_str());
      return 1;
    }
    if (expected.size() != result.size()) {
      fprintf(stderr, "Error: expected %zu elements, got %zu\n",
              expected.size(), result.size());
      return 1;
    }
    errors = verify(result.data(), expected.data(), result.size(), args.atol, args.rtol);
  }

  if (args.output.empty() && args.expectedOutput.empty()) {
    printf("Result: ");
    for (size_t i = 0; i < outBuf.shape.size(); i++) {
      if (i > 0) printf("x");
      printf("%ld", outBuf.shape[i]);
    }
    printf("x%s\n", outBuf.dtype.c_str());
    float minV = result[0], maxV = result[0];
    for (size_t i = 1; i < result.size(); i++) {
      if (result[i] < minV) minV = result[i];
      if (result[i] > maxV) maxV = result[i];
    }
    printf("  min=%f, max=%f, elements=%zu\n", minV, maxV, result.size());
  } else if (args.output == "-") {
    for (size_t i = 0; i < result.size() && i < 20; i++)
      printf("  [%zu] = %f\n", i, result[i]);
    if (result.size() > 20)
      printf("  ... (%zu more elements)\n", result.size() - 20);
  }

  // Cleanup.
  vkDestroyFence(device, fence, nullptr);
  vkDestroyCommandPool(device, cmdPool, nullptr);
  vkDestroyDescriptorPool(device, descPool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, dsLayout, nullptr);
  vkDestroyShaderModule(device, shaderModule, nullptr);
  for (auto &vb : vkBufs) {
    vkDestroyBuffer(device, vb.buffer, nullptr);
    vkFreeMemory(device, vb.memory, nullptr);
  }
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  if (vkLib) dlclose(vkLib);
  return errors > 0 ? 1 : 0;
}
