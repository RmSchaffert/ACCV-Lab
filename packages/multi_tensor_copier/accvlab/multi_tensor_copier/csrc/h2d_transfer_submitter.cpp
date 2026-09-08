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

#include "h2d_transfer_submitter.h"

#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace accvlab::multi_tensor_copier::internal {

std::vector<H2DTransferChunk> plan_h2d_transfer_chunks(const at::Tensor& destination,
                                                       const at::Tensor& source, int64_t max_chunk_bytes) {
    if (max_chunk_bytes <= 0 || !source.device().is_cpu() || !destination.device().is_cuda() ||
        !source.is_contiguous() || !destination.is_contiguous() ||
        source.scalar_type() != destination.scalar_type() || source.numel() != destination.numel() ||
        source.numel() == 0) {
        return {};
    }

    const int64_t element_size = static_cast<int64_t>(source.element_size());
    const int64_t max_chunk_elements = std::max<int64_t>(1, max_chunk_bytes / element_size);
    std::vector<H2DTransferChunk> chunks;
    chunks.reserve(static_cast<size_t>((source.numel() + max_chunk_elements - 1) / max_chunk_elements));
    for (int64_t offset = 0; offset < source.numel(); offset += max_chunk_elements) {
        chunks.push_back(H2DTransferChunk{
            offset,
            std::min(max_chunk_elements, source.numel() - offset),
        });
    }
    return chunks;
}

void H2DTransferSubmitter::submit(at::Tensor destination, const at::Tensor& source, bool non_blocking) {
    const auto chunks = plan_h2d_transfer_chunks(destination, source, max_chunk_bytes_);
    if (chunks.empty()) {
        destination.copy_(source, non_blocking);
        return;
    }
    if (!target_stream_.has_value()) {
        throw std::runtime_error("Internal error: paced H2D transfer has no target CUDA stream");
    }

    const auto stream = *target_stream_;
    c10::cuda::CUDAGuard guard(stream.device_index());
    at::cuda::CUDAStreamGuard stream_guard(stream);
    const auto flat_source = source.view({-1});
    const auto flat_destination = destination.view({-1});

    for (const auto& chunk : chunks) {
        wait_for_in_flight_chunk();
        flat_destination.narrow(0, chunk.element_offset, chunk.element_count)
            .copy_(flat_source.narrow(0, chunk.element_offset, chunk.element_count), non_blocking);
        record_in_flight_chunk();
    }
}

void H2DTransferSubmitter::wait_for_in_flight_chunk() {
    if (!in_flight_chunk_.has_value() || in_flight_chunk_->ev == nullptr) {
        return;
    }
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(in_flight_chunk_->device_index));
    const auto status = cudaEventSynchronize(in_flight_chunk_->ev);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaEventSynchronize failed while pacing H2D transfers: ") +
                                 cudaGetErrorString(status));
    }
}

void H2DTransferSubmitter::record_in_flight_chunk() {
    const auto stream = *target_stream_;
    c10::cuda::CUDAGuard guard(stream.device_index());
    if (!in_flight_chunk_.has_value()) {
        cudaEvent_t event = nullptr;
        const auto status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaEventCreateWithFlags failed while pacing H2D transfers: ") +
                cudaGetErrorString(status));
        }
        in_flight_chunk_.emplace(event, static_cast<int>(stream.device_index()));
    }

    const auto status = cudaEventRecord(in_flight_chunk_->ev, stream.stream());
    if (status != cudaSuccess) {
        in_flight_chunk_.reset();
        throw std::runtime_error(std::string("cudaEventRecord failed while pacing H2D transfers: ") +
                                 cudaGetErrorString(status));
    }
}

}  // namespace accvlab::multi_tensor_copier::internal
