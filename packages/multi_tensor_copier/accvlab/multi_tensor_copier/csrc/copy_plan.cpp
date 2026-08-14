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

#include "copy_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace accvlab::multi_tensor_copier::internal {
namespace {

int64_t round_up_i64(int64_t x, int64_t a) {
    if (a <= 1) {
        return x;
    }
    const int64_t rem = x % a;
    const int64_t res = rem == 0 ? x : (x + (a - rem));
    return res;
}

int64_t next_pow2_i64(int64_t x) {
    if (x <= 1) {
        return 1;
    }
    // Round up to the next power of two (clamped to int64 range).
    uint64_t v = static_cast<uint64_t>(x - 1);
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v += 1;
    if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return std::numeric_limits<int64_t>::max();
    }
    return static_cast<int64_t>(v);
}

// Bucket ordering key: clamp a required alignment to {16,8,4,2,1} by rounding down to the nearest bucket <= 16.
// IMPORTANT: This is only used for ordering; the actual per-tensor alignment is preserved separately.
int64_t pack_bucket_key(int64_t required_align) {
    if (required_align >= 16) {
        return 16;
    }
    if (required_align >= 8) {
        return 8;
    }
    if (required_align >= 4) {
        return 4;
    }
    if (required_align >= 2) {
        return 2;
    }
    return 1;
}

// Packing candidate: an input tensor i that can be packed into the CPU staging byte buffer.
struct PackCandidate {
    size_t idx;
    int64_t bytes;
    int64_t required_align;
};

// Buckets of packing candidates in descending alignment order.
// Note: complex128 element_size() is 16, complex64 is 8.
struct PackBuckets {
    std::vector<PackCandidate> a16;
    std::vector<PackCandidate> a8;
    std::vector<PackCandidate> a4;
    std::vector<PackCandidate> a2;
    std::vector<PackCandidate> a1;

    void add(PackCandidate c) {
        switch (pack_bucket_key(c.required_align)) {
            case 16:
                a16.push_back(c);
                break;
            case 8:
                a8.push_back(c);
                break;
            case 4:
                a4.push_back(c);
                break;
            case 2:
                a2.push_back(c);
                break;
            default:
                a1.push_back(c);
                break;
        }
    }

    template <typename F>
    void for_each_bucket_desc(F&& f) const {
        f(16, a16);
        f(8, a8);
        f(4, a4);
        f(2, a2);
        f(1, a1);
    }
};

std::optional<PackCandidate> make_pack_candidate(const std::vector<at::Tensor>& inputs,
                                                 const c10::Device& target_device, size_t i,
                                                 int64_t min_align) {
    // Heuristic thresholds: only pack "small" tensors.
    constexpr int64_t kPackMaxBytesPerTensor = 256 * 1024;  // 256KB

    const auto& in = inputs[i];
    // Only consider CPU tensors that will be transferred to CUDA.
    if (!in.device().is_cpu() || in.device() == target_device) {
        return std::nullopt;
    }
    // Packing requires a flat contiguous view.
    if (!in.is_contiguous()) {
        return std::nullopt;
    }
    const int64_t bytes = in.numel() * in.element_size();
    // Skip tensors that are too big; packing targets "many tiny tensors" overhead.
    if (bytes == 0 || bytes > kPackMaxBytesPerTensor) {
        return std::nullopt;
    }
    const int64_t elem_sz = static_cast<int64_t>(in.element_size());
    // Effective alignment must be >= requested minimum AND must guarantee element alignment.
    // If min_align is not a multiple of elem_sz, round up to the next multiple to preserve
    // the invariant that byte_offset % elem_sz == 0.
    int64_t required_align = std::max<int64_t>(min_align, elem_sz);
    required_align = round_up_i64(required_align, elem_sz);
    return PackCandidate{i, bytes, required_align};
}

// Assign byte offsets within chunked packed buffers for each candidate tensor, processing
// alignment buckets in descending order to minimise inter-tensor padding.  When a tensor
// would exceed `max_chunk_bytes` in the current chunk, a new chunk is started.  Populates
// `pack_plan.byte_offset_by_input`, `chunk_index_by_input`, and `chunk_sizes`.
void layout_packed_offsets(const PackBuckets& buckets, PackPlan& pack_plan, int64_t& packed_count,
                           int64_t max_chunk_bytes) {
    int64_t cursor = 0;
    int64_t chunk_idx = 0;
    packed_count = 0;

    auto finalize_chunk = [&]() {
        if (cursor > 0) {
            pack_plan.chunk_sizes.push_back(cursor);
            cursor = 0;
            ++chunk_idx;
        }
    };

    auto pack_bucket = [&](int64_t bucket_align, const std::vector<PackCandidate>& bucket) {
        if (bucket.empty()) {
            return;
        }
        for (const auto& c : bucket) {
            int64_t aligned_cursor = round_up_i64(cursor, c.required_align);
            if (aligned_cursor + c.bytes > max_chunk_bytes && cursor > 0) {
                finalize_chunk();
                aligned_cursor = round_up_i64(cursor, c.required_align);
            }
            cursor = aligned_cursor;
            pack_plan.byte_offset_by_input[c.idx] = cursor;
            pack_plan.chunk_index_by_input[c.idx] = chunk_idx;
            cursor += c.bytes;
            packed_count += 1;
        }
    };

    buckets.for_each_bucket_desc(pack_bucket);
    if (cursor > 0) {
        pack_plan.chunk_sizes.push_back(cursor);
    }
}

}  // namespace

int64_t packed_buffer_alignment_bytes(int64_t min_packed_alignment_bytes) {
    // Ensure the packed buffer itself is aligned to at least this many bytes.
    // Also round up to a power-of-two so we can use bit-masking for pointer alignment.
    return next_pow2_i64(std::max<int64_t>(16, min_packed_alignment_bytes));
}

int64_t aligned_slice_offset_bytes(const void* base_ptr, int64_t alignment_pow2) {
    if (alignment_pow2 <= 1) {
        return 0;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(base_ptr);
    const uintptr_t alignment = static_cast<uintptr_t>(alignment_pow2);
    const uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
    return static_cast<int64_t>(aligned - base);
}

// Decide whether to enable the packed-CPU-tensors fast path and, if enabled, compute
// per-tensor chunk assignments and byte offsets within each chunk.
PackPlan compute_pack_plan(const std::vector<at::Tensor>& inputs, const c10::Device& target_device,
                           bool pack_cpu_tensors, int64_t min_packed_alignment_bytes,
                           int64_t max_packed_chunk_bytes) {
    PackPlan pack_plan;
    pack_plan.byte_offset_by_input.assign(inputs.size(), -1);
    pack_plan.chunk_index_by_input.assign(inputs.size(), -1);

    if (!pack_cpu_tensors || !target_device.is_cuda()) {
        return pack_plan;
    }

    // We only pack tensors that:
    // - are on CPU (and not already on the target device),
    // - are contiguous (so we can treat them as a flat buffer),
    // - are "small enough" individually,
    //
    // Mixed-dtype packing: we pack raw bytes and reconstruct typed tensors as views
    // sharing the packed GPU storage.
    const int64_t min_align = std::max<int64_t>(1, min_packed_alignment_bytes);
    PackBuckets buckets;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (auto cand = make_pack_candidate(inputs, target_device, i, min_align)) {
            buckets.add(*cand);
        }
    }

    int64_t packed_count = 0;
    layout_packed_offsets(buckets, pack_plan, packed_count, max_packed_chunk_bytes);
    if (packed_count >= 2 && !pack_plan.chunk_sizes.empty()) {
        pack_plan.enabled = true;
    } else {
        pack_plan.enabled = false;
        pack_plan.chunk_sizes.clear();
        std::fill(pack_plan.byte_offset_by_input.begin(), pack_plan.byte_offset_by_input.end(), -1);
        std::fill(pack_plan.chunk_index_by_input.begin(), pack_plan.chunk_index_by_input.end(), -1);
    }
    return pack_plan;
}

}  // namespace accvlab::multi_tensor_copier::internal
