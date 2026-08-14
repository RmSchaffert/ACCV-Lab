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

#include <ATen/ATen.h>
#include <c10/core/Device.h>

#include <cstdint>
#include <vector>

namespace accvlab::multi_tensor_copier::internal {

// A precomputed plan for the optional "pack many small CPU tensors into one staging buffer" fast path.
// When enabled, multiple small contiguous CPU tensors (mix of different dtypes allowed) are copied into a
// single packed *byte* buffer (pinned or pageable), transferred with a single H2D, and then reconstructed as
// per-tensor views sharing the packed GPU storage.
//
// For each input i: byte_offset_by_input[i] is the starting *byte* offset inside the packed buffer,
// or -1 if this input is not packed.
struct PackPlan {
    // For each input leaf i: starting byte offset inside its chunk, or -1 if not packed.
    // IMPORTANT: This is checked for all tensors, not only the packed ones, and not only if packing is
    //            enabled. Therefore, it has to be initialized with -1 for all non-packed inputs.
    std::vector<int64_t> byte_offset_by_input;
    // For each input leaf i: which chunk it belongs to, or -1 if not packed.
    std::vector<int64_t> chunk_index_by_input;
    // Byte size of each chunk (one entry per chunk).
    std::vector<int64_t> chunk_sizes;
    // Whether packing is enabled for this call (if false, treat everything as "not packed").
    bool enabled{false};
};

int64_t packed_buffer_alignment_bytes(int64_t min_packed_alignment_bytes);

int64_t aligned_slice_offset_bytes(const void* base_ptr, int64_t alignment_pow2);

PackPlan compute_pack_plan(const std::vector<at::Tensor>& inputs, const c10::Device& target_device,
                           bool pack_cpu_tensors, int64_t min_packed_alignment_bytes,
                           int64_t max_packed_chunk_bytes);

}  // namespace accvlab::multi_tensor_copier::internal
