################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# runtime/src/iree/builtins/ukernel/arch/arm_64/BUILD.bazel                    #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_compiler_targeting_iree_arch(_IREE_UKERNEL_BITCODE_BUILD_ARM_64 "arm_64")
if(_IREE_UKERNEL_BITCODE_BUILD_ARM_64)

iree_bitcode_library(
  NAME
    ukernel_bitcode_arch_arm_64_entry_points
  ARCH
    arm_64
  INTERNAL_HDRS
    "${PROJECT_BINARY_DIR}/runtime/src/iree/builtins/ukernel/internal_headers_filegroup.stamp"
    "${PROJECT_BINARY_DIR}/runtime/src/iree/schemas/cpu_data_headers_filegroup.stamp"
    "common_arm_64.h"
    "mmt4d_arm_64_internal.h"
    "mmt4d_arm_64_tiles.inl"
    "pack_arm_64_internal.h"
    "unpack_arm_64_internal.h"
  SRCS
    "mmt4d_arm_64_entry_point.c"
    "pack_arm_64_entry_point.c"
    "unpack_arm_64_entry_point.c"
)

iree_bitcode_library(
  NAME
    ukernel_bitcode_arch_arm_64_base
  ARCH
    arm_64
  INTERNAL_HDRS
    "${PROJECT_BINARY_DIR}/runtime/src/iree/builtins/ukernel/internal_headers_filegroup.stamp"
    "${PROJECT_BINARY_DIR}/runtime/src/iree/schemas/cpu_data_headers_filegroup.stamp"
    "common_arm_64.h"
    "mmt4d_arm_64_internal.h"
    "mmt4d_arm_64_tiles.inl"
    "pack_arm_64_internal.h"
    "unpack_arm_64_internal.h"
  SRCS
    "mmt4d_arm_64_base.c"
    "pack_arm_64_base.c"
    "unpack_arm_64_base.c"
)

iree_bitcode_library(
  NAME
    ukernel_bitcode_arch_arm_64_fullfp16
  ARCH
    arm_64
  INTERNAL_HDRS
    "${PROJECT_BINARY_DIR}/runtime/src/iree/builtins/ukernel/internal_headers_filegroup.stamp"
    "${PROJECT_BINARY_DIR}/runtime/src/iree/schemas/cpu_data_headers_filegroup.stamp"
    "common_arm_64.h"
    "mmt4d_arm_64_internal.h"
    "mmt4d_arm_64_tiles.inl"
    "pack_arm_64_internal.h"
    "unpack_arm_64_internal.h"
  SRCS
    "mmt4d_arm_64_fullfp16.c"
  COPTS
    "-march=armv8.2-a+fp16"
)

iree_bitcode_library(
  NAME
    ukernel_bitcode_arch_arm_64_fp16fml
  ARCH
    arm_64
  INTERNAL_HDRS
    "${PROJECT_BINARY_DIR}/runtime/src/iree/builtins/ukernel/internal_headers_filegroup.stamp"
    "${PROJECT_BINARY_DIR}/runtime/src/iree/schemas/cpu_data_headers_filegroup.stamp"
    "common_arm_64.h"
    "mmt4d_arm_64_internal.h"
    "mmt4d_arm_64_tiles.inl"
    "pack_arm_64_internal.h"
    "unpack_arm_64_internal.h"
  SRCS
    "mmt4d_arm_64_fp16fml.c"
  COPTS
    "-march=armv8.2-a+fp16fml"
)

iree_bitcode_library(
  NAME
    ukernel_bitcode_arch_arm_64_bf16
  ARCH
    arm_64
  INTERNAL_HDRS
    "${PROJECT_BINARY_DIR}/runtime/src/iree/builtins/ukernel/internal_headers_filegroup.stamp"
    "${PROJECT_BINARY_DIR}/runtime/src/iree/schemas/cpu_data_headers_filegroup.stamp"
    "common_arm_64.h"
    "mmt4d_arm_64_internal.h"
    "mmt4d_arm_64_tiles.inl"
    "pack_arm_64_internal.h"
    "unpack_arm_64_internal.h"
  SRCS
    "mmt4d_arm_64_bf16.c"
  COPTS
    "-march=armv8.2-a+bf16"
)

iree_bitcode_library(
  NAME
    ukernel_bitcode_arch_arm_64_dotprod
  ARCH
    arm_64
  INTERNAL_HDRS
    "${PROJECT_BINARY_DIR}/runtime/src/iree/builtins/ukernel/internal_headers_filegroup.stamp"
    "${PROJECT_BINARY_DIR}/runtime/src/iree/schemas/cpu_data_headers_filegroup.stamp"
    "common_arm_64.h"
    "mmt4d_arm_64_internal.h"
    "mmt4d_arm_64_tiles.inl"
    "pack_arm_64_internal.h"
    "unpack_arm_64_internal.h"
  SRCS
    "mmt4d_arm_64_dotprod.c"
  COPTS
    "-march=armv8.2-a+dotprod"
)

iree_bitcode_library(
  NAME
    ukernel_bitcode_arch_arm_64_i8mm
  ARCH
    arm_64
  INTERNAL_HDRS
    "${PROJECT_BINARY_DIR}/runtime/src/iree/builtins/ukernel/internal_headers_filegroup.stamp"
    "${PROJECT_BINARY_DIR}/runtime/src/iree/schemas/cpu_data_headers_filegroup.stamp"
    "common_arm_64.h"
    "mmt4d_arm_64_internal.h"
    "mmt4d_arm_64_tiles.inl"
    "pack_arm_64_internal.h"
    "unpack_arm_64_internal.h"
  SRCS
    "mmt4d_arm_64_i8mm.c"
  COPTS
    "-march=armv8.2-a+i8mm"
)

iree_link_bitcode(
  NAME
    ukernel_bitcode_arch_arm_64
  SRCS
    "ukernel_bitcode_arch_arm_64_base.bc"
    "ukernel_bitcode_arch_arm_64_bf16.bc"
    "ukernel_bitcode_arch_arm_64_dotprod.bc"
    "ukernel_bitcode_arch_arm_64_entry_points.bc"
    "ukernel_bitcode_arch_arm_64_fp16fml.bc"
    "ukernel_bitcode_arch_arm_64_fullfp16.bc"
    "ukernel_bitcode_arch_arm_64_i8mm.bc"

)

elseif(IREE_BUILD_COMPILER AND IREE_TARGET_BACKEND_LLVM_CPU)
iree_make_empty_file("${CMAKE_CURRENT_BINARY_DIR}/ukernel_bitcode_arch_arm_64.bc")
endif()  # _IREE_UKERNEL_BITCODE_BUILD_ARM_64

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###

if (NOT (IREE_ARCH STREQUAL "arm_64"))
  return()
endif()

iree_select_compiler_opts(IREE_UK_COPTS_ARM_64_FULLFP16
  CLANG_OR_GCC
    "-march=armv8.2-a+fp16"
)

iree_select_compiler_opts(IREE_UK_COPTS_ARM_64_FP16FML
  CLANG_OR_GCC
    "-march=armv8.2-a+fp16fml"
)

iree_select_compiler_opts(IREE_UK_COPTS_ARM_64_BF16
  CLANG_OR_GCC
    "-march=armv8.2-a+bf16"
)

iree_select_compiler_opts(IREE_UK_COPTS_ARM_64_DOTPROD
  CLANG_OR_GCC
    "-march=armv8.2-a+dotprod"
)

iree_select_compiler_opts(IREE_UK_COPTS_ARM_64_I8MM
  CLANG_OR_GCC
    "-march=armv8.2-a+i8mm"
)

check_cxx_compiler_flag("${IREE_UK_COPTS_ARM_64_FULLFP16}" IREE_UK_BUILD_ARM_64_FULLFP16)
check_cxx_compiler_flag("${IREE_UK_COPTS_ARM_64_FP16FML}" IREE_UK_BUILD_ARM_64_FP16FML)
check_cxx_compiler_flag("${IREE_UK_COPTS_ARM_64_BF16}" IREE_UK_BUILD_ARM_64_BF16)
check_cxx_compiler_flag("${IREE_UK_COPTS_ARM_64_DOTPROD}" IREE_UK_BUILD_ARM_64_DOTPROD)
check_cxx_compiler_flag("${IREE_UK_COPTS_ARM_64_I8MM}" IREE_UK_BUILD_ARM_64_I8MM)
configure_file("config_arm_64.h.in" "config_arm_64.h")

iree_cc_library(
  NAME
    common_arm_64
  HDRS
    "common_arm_64.h"
  DEPS
    iree::builtins::ukernel::internal_headers
    iree::schemas::cpu_data
)

set(IREE_UK_ARM_64_DEPS "")

if(IREE_UK_BUILD_ARM_64_FULLFP16)
iree_cc_library(
  NAME
    arm_64_fullfp16
  SRCS
    "mmt4d_arm_64_fullfp16.c"
  COPTS
    "${IREE_UK_COPTS_ARM_64_FULLFP16}"
  DEPS
    iree::builtins::ukernel::internal_headers
)
list(APPEND IREE_UK_ARM_64_DEPS "::arm_64_fullfp16")
endif()  # IREE_UK_BUILD_ARM_64_FULLFP16

if(IREE_UK_BUILD_ARM_64_FP16FML)
iree_cc_library(
  NAME
    arm_64_fp16fml
  SRCS
    "mmt4d_arm_64_fp16fml.c"
  COPTS
    "${IREE_UK_COPTS_ARM_64_FP16FML}"
  DEPS
    iree::builtins::ukernel::internal_headers
)
list(APPEND IREE_UK_ARM_64_DEPS "::arm_64_fp16fml")
endif()  # IREE_UK_BUILD_ARM_64_FP16FML

if(IREE_UK_BUILD_ARM_64_BF16)
iree_cc_library(
  NAME
    arm_64_bf16
  SRCS
    "mmt4d_arm_64_bf16.c"
  COPTS
    "${IREE_UK_COPTS_ARM_64_BF16}"
  DEPS
    iree::builtins::ukernel::internal_headers
)
list(APPEND IREE_UK_ARM_64_DEPS "::arm_64_bf16")
endif()  # IREE_UK_BUILD_ARM_64_BF16

if(IREE_UK_BUILD_ARM_64_DOTPROD)
iree_cc_library(
  NAME
    arm_64_dotprod
  SRCS
    "mmt4d_arm_64_dotprod.c"
  COPTS
    "${IREE_UK_COPTS_ARM_64_DOTPROD}"
  DEPS
    iree::builtins::ukernel::internal_headers
)
list(APPEND IREE_UK_ARM_64_DEPS "::arm_64_dotprod")
endif()  # IREE_UK_BUILD_ARM_64_DOTPROD

if(IREE_UK_BUILD_ARM_64_I8MM)
iree_cc_library(
  NAME
    arm_64_i8mm
  SRCS
    "mmt4d_arm_64_i8mm.c"
  COPTS
    "${IREE_UK_COPTS_ARM_64_I8MM}"
  DEPS
    iree::builtins::ukernel::internal_headers
)
list(APPEND IREE_UK_ARM_64_DEPS "::arm_64_i8mm")
endif()  # IREE_UK_BUILD_ARM_64_I8MM

iree_cc_library(
  NAME
    arm_64
  SRCS
    "mmt4d_arm_64_entry_point.c"
    "mmt4d_arm_64_base.c"
    "pack_arm_64_entry_point.c"
    "pack_arm_64_base.c"
    "query_tile_sizes_arm_64_entry_point.c"
    "unpack_arm_64_entry_point.c"
    "unpack_arm_64_base.c"
  DEPS
    ::common_arm_64
    iree::base::core_headers
    iree::schemas::cpu_data
    iree::builtins::ukernel::internal_headers
    ${IREE_UK_ARM_64_DEPS}
  PUBLIC
)

set(IREE_UK_ARCH_DEPS "iree::builtins::ukernel::arch::arm_64" PARENT_SCOPE)
