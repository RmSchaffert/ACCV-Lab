/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LANE_HELPERS_EXT_IMPL_HELPER_MACROS_CUH
#define LANE_HELPERS_EXT_IMPL_HELPER_MACROS_CUH

#include <stdexcept>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(error_code_or_call) C10_CUDA_CHECK(error_code_or_call)
#define CUDA_CHECK_LAST() C10_CUDA_CHECK(cudaGetLastError())

namespace lane_helpers::ext_impl {

inline void check_non_negative_cuda_device(int device) {
    if (device < 0) {
        throw std::runtime_error("CUDA device index must be non-negative.");
    }
}

}  // namespace lane_helpers::ext_impl

#endif  // LANE_HELPERS_EXT_IMPL_HELPER_MACROS_CUH
