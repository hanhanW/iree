// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_BUFFER_H_
#define IREE_HAL_DRIVERS_WEBGPU_BUFFER_H_

#include "experimental/webgpu/platform/webgpu.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TODO(benvanik): callback for releasing the buffer back to the allocator if
// we start to support pooling.

iree_status_t iree_hal_webgpu_buffer_wrap(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    WGPUBuffer handle, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

WGPUBuffer iree_hal_webgpu_buffer_handle(const iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_BUFFER_H_
