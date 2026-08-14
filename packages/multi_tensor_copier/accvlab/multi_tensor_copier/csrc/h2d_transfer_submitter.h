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

#include "cuda_event.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace accvlab::multi_tensor_copier::internal {

struct H2DTransferChunk {
    int64_t element_offset;
    int64_t element_count;
};

std::vector<H2DTransferChunk> plan_h2d_transfer_chunks(const at::Tensor& destination,
                                                       const at::Tensor& source, int64_t max_chunk_bytes);

class H2DTransferSubmitter {
   public:
    H2DTransferSubmitter(std::optional<at::cuda::CUDAStream> target_stream, int64_t max_chunk_bytes)
        : target_stream_(target_stream), max_chunk_bytes_(max_chunk_bytes) {}

    void submit(at::Tensor destination, const at::Tensor& source, bool non_blocking);

   private:
    void wait_for_in_flight_chunk();
    void record_in_flight_chunk();

    std::optional<at::cuda::CUDAStream> target_stream_;
    int64_t max_chunk_bytes_{0};
    std::optional<CudaEvent> in_flight_chunk_;
};

}  // namespace accvlab::multi_tensor_copier::internal
