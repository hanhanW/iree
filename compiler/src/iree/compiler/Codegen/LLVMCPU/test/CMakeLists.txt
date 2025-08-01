################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# compiler/src/iree/compiler/Codegen/LLVMCPU/test/BUILD.bazel                  #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "2d_scalable_to_1d_scalable.mlir"
    "aarch64_dotprod_vector_lowering.mlir"
    "aarch64_vector_lowering.mlir"
    "apply_scale_lowering.mlir"
    "assign_constant_ordinals.mlir"
    "assign_import_ordinals.mlir"
    "check_ir_before_llvm_conversion.mlir"
    "check_ir_before_llvm_conversion_not_fail_unbound.mlir"
    "convert_to_llvm.mlir"
    "emit_vectorization_remarks.mlir"
    "expand_f16_op_to_f32.mlir"
    "hal_executable_constants.mlir"
    "hal_interface_bindings.mlir"
    "hal_interface_constants.mlir"
    "hal_interface_workgroup_info.mlir"
    "illegal_configuration.mlir"
    "peel.mlir"
    "pipeline_arm_sme_streaming_mode_tests.mlir"
    "pipeline_disable_distribution_tests.mlir"
    "pipeline_pack_unpack_tests.mlir"
    "pipeline_pad_conv_tests.mlir"
    "pipeline_pad_tests.mlir"
    "pipeline_peel_and_vectorize_tests.mlir"
    "pipeline_riscv_aggressive_distribution_tests.mlir"
    "pipeline_split_reduction_tests.mlir"
    "pipeline_tests.mlir"
    "pipeline_transpose_avx2_tests.mlir"
    "pipeline_vector_masking_tests.mlir"
    "pipeline_vectorize_nd_extract_tests.mlir"
    "scalable_tile_and_vectorize_matmul.mlir"
    "select_aarch64_lowering_strategy.mlir"
    "select_aarch64_sme_lowering_strategy.mlir"
    "select_aarch64_sve_lowering_strategy.mlir"
    "select_aarch64_sve_lowering_strategy_peeling.mlir"
    "select_lowering_strategy_without_distribution.mlir"
    "select_riscv_lowering_strategy.mlir"
    "select_x86_64_lowering_strategy.mlir"
    "split_reduction.mlir"
    "synchronize_symbol_visibility.mlir"
    "tile.mlir"
    "tile_and_fuse.mlir"
    "tile_root_and_fuse_producer_consumer.mlir"
    "unfused_fma.mlir"
    "vector_contract_to_arm_asm.mlir"
    "vector_contract_to_arm_intrinsics.mlir"
    "vector_lowering.mlir"
    "vector_transpose_lowering.mlir"
    "vectorize_with_masking_and_hoist.mlir"
    "verify_linalg_transform_legality.mlir"
    "verify_vector_size_legality.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-opt
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
