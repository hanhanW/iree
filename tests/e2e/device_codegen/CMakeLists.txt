# Device-codegen-only e2e tests.
#
# Full flow:
#   1. iree-compile + iree-run-module on CPU to generate reference output
#   2. iree-device-codegen compiles func.func -> HSACO
#   3. run-hip-kernel launches on GPU and verifies against reference
#
# Requirements:
#   - IREE_TARGET_BACKEND_ROCM: ROCM codegen backend is built
#   - IREE_HAL_DRIVER_HIP: HIP runtime is available
#   - run-hip-kernel binary is built (requires HIP SDK)

if(NOT IREE_TARGET_BACKEND_ROCM)
  return()
endif()

if(NOT IREE_HAL_DRIVER_HIP)
  return()
endif()

if(NOT TARGET run-hip-kernel)
  message(STATUS "device_codegen e2e tests: skipped (run-hip-kernel not built)")
  return()
endif()

iree_lit_test_suite(
  NAME
    e2e
  SRCS
    "matmul_bias.mlir"
    "matmul_bias_dynamic.mlir"
    "matmul_bias_fully_dynamic.mlir"
  TOOLS
    iree-compile
    iree-device-codegen
    iree-run-module
    run-hip-kernel
    FileCheck
  LABELS
    "driver=hip"
    "requires-gpu-rocm"
)
