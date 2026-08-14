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

#pragma once

#include <cuda_runtime.h>

namespace accvlab::multi_tensor_copier::internal {

// Minimal RAII wrapper around a CUDA event.
//
// Events are used to track completion of all enqueued copies on a CUDA stream,
// so `ready()` can poll and `get()` / destructor can wait before releasing
// staging buffers.
struct CudaEvent {
    cudaEvent_t ev{nullptr};
    int device_index{-1};

    CudaEvent() = default;
    CudaEvent(cudaEvent_t e, int dev) : ev(e), device_index(dev) {}

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept {
        ev = other.ev;
        device_index = other.device_index;
        other.ev = nullptr;
        other.device_index = -1;
    }
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            cleanup_no_throw();
            ev = other.ev;
            device_index = other.device_index;
            other.ev = nullptr;
            other.device_index = -1;
        }
        return *this;
    }

    ~CudaEvent() { cleanup_no_throw(); }

    void cleanup_no_throw() noexcept {
        if (ev != nullptr) {
            // cudaEventDestroy does not require the Python GIL.
            cudaEventDestroy(ev);
            ev = nullptr;
            device_index = -1;
        }
    }
};

}  // namespace accvlab::multi_tensor_copier::internal
