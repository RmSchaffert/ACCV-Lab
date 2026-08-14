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

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/core/Device.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using accvlab::multi_tensor_copier::internal::aligned_slice_offset_bytes;
using accvlab::multi_tensor_copier::internal::compute_pack_plan;
using accvlab::multi_tensor_copier::internal::packed_buffer_alignment_bytes;
using accvlab::multi_tensor_copier::internal::PackPlan;

const c10::Device kCudaTarget(c10::kCUDA, 0);

TEST(CopyPlanTest, DisabledPackingInitializesFallbackEntries) {
    const std::vector<at::Tensor> inputs = {
        at::zeros({8}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU)),
        at::zeros({8}, at::TensorOptions().dtype(at::kLong).device(at::kCPU)),
    };

    const PackPlan plan = compute_pack_plan(inputs, kCudaTarget, false, 16, 1024);

    EXPECT_FALSE(plan.enabled);
    EXPECT_EQ(plan.byte_offset_by_input, std::vector<int64_t>({-1, -1}));
    EXPECT_EQ(plan.chunk_index_by_input, std::vector<int64_t>({-1, -1}));
    EXPECT_TRUE(plan.chunk_sizes.empty());
}

TEST(CopyPlanTest, RequiresAtLeastTwoPackingCandidates) {
    const std::vector<at::Tensor> inputs = {
        at::zeros({8}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU)),
    };

    const PackPlan plan = compute_pack_plan(inputs, kCudaTarget, true, 16, 1024);

    EXPECT_FALSE(plan.enabled);
    EXPECT_EQ(plan.byte_offset_by_input, std::vector<int64_t>({-1}));
    EXPECT_EQ(plan.chunk_index_by_input, std::vector<int64_t>({-1}));
    EXPECT_TRUE(plan.chunk_sizes.empty());
}

TEST(CopyPlanTest, FiltersIneligibleTensors) {
    const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    const at::Tensor first_small = at::zeros({8}, options);
    const at::Tensor non_contiguous = at::zeros({3, 4}, options).transpose(0, 1);
    const at::Tensor too_large = at::zeros({256 * 1024 / 4 + 1}, options);
    const at::Tensor empty = at::zeros({0}, options);
    const at::Tensor second_small = at::zeros({4}, options);
    const std::vector<at::Tensor> inputs = {
        first_small, non_contiguous, too_large, empty, second_small,
    };

    const PackPlan plan = compute_pack_plan(inputs, kCudaTarget, true, 16, 1024);

    ASSERT_TRUE(plan.enabled);
    EXPECT_GE(plan.byte_offset_by_input[0], 0);
    EXPECT_EQ(plan.byte_offset_by_input[1], -1);
    EXPECT_EQ(plan.byte_offset_by_input[2], -1);
    EXPECT_EQ(plan.byte_offset_by_input[3], -1);
    EXPECT_GE(plan.byte_offset_by_input[4], 0);
}

TEST(CopyPlanTest, SplitsCandidatesAtPackedChunkLimit) {
    const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    const std::vector<at::Tensor> inputs = {
        at::zeros({64}, options),
        at::zeros({64}, options),
        at::zeros({64}, options),
        at::zeros({64}, options),
    };

    const PackPlan plan = compute_pack_plan(inputs, kCudaTarget, true, 16, 512);

    ASSERT_TRUE(plan.enabled);
    EXPECT_EQ(plan.byte_offset_by_input, std::vector<int64_t>({0, 256, 0, 256}));
    EXPECT_EQ(plan.chunk_index_by_input, std::vector<int64_t>({0, 0, 1, 1}));
    EXPECT_EQ(plan.chunk_sizes, std::vector<int64_t>({512, 512}));
}

TEST(CopyPlanTest, PreservesPerTensorAlignmentAcrossMixedDtypes) {
    const std::vector<at::Tensor> inputs = {
        at::zeros({3}, at::TensorOptions().dtype(at::kByte).device(at::kCPU)),
        at::zeros({3}, at::TensorOptions().dtype(at::kShort).device(at::kCPU)),
        at::zeros({3}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU)),
        at::zeros({3}, at::TensorOptions().dtype(at::kDouble).device(at::kCPU)),
        at::zeros({3}, at::TensorOptions().dtype(at::kComplexDouble).device(at::kCPU)),
    };

    const PackPlan plan = compute_pack_plan(inputs, kCudaTarget, true, 6, 1024);

    ASSERT_TRUE(plan.enabled);
    ASSERT_EQ(plan.chunk_sizes.size(), 1);
    for (size_t index = 0; index < inputs.size(); ++index) {
        const int64_t element_size = static_cast<int64_t>(inputs[index].element_size());
        const int64_t required_alignment =
            ((std::max<int64_t>(6, element_size) + element_size - 1) / element_size) * element_size;
        EXPECT_EQ(plan.byte_offset_by_input[index] % required_alignment, 0);
    }
}

TEST(CopyPlanTest, IncludesTensorAtPackingThreshold) {
    const std::vector<at::Tensor> inputs = {
        at::zeros({256 * 1024}, at::TensorOptions().dtype(at::kByte).device(at::kCPU)),
        at::zeros({1}, at::TensorOptions().dtype(at::kByte).device(at::kCPU)),
    };

    const PackPlan plan = compute_pack_plan(inputs, kCudaTarget, true, 16, 512 * 1024);

    ASSERT_TRUE(plan.enabled);
    EXPECT_GE(plan.byte_offset_by_input[0], 0);
    EXPECT_GE(plan.byte_offset_by_input[1], 0);
}

TEST(CopyPlanTest, ComputesPackedBufferAlignmentAndSliceOffset) {
    EXPECT_EQ(packed_buffer_alignment_bytes(1), 16);
    EXPECT_EQ(packed_buffer_alignment_bytes(16), 16);
    EXPECT_EQ(packed_buffer_alignment_bytes(17), 32);

    std::array<std::byte, 64> storage{};
    const auto base = reinterpret_cast<uintptr_t>(storage.data());
    const int64_t expected_offset = static_cast<int64_t>(((base + 15) & ~uintptr_t{15}) - base);
    EXPECT_EQ(aligned_slice_offset_bytes(storage.data(), 16), expected_offset);
}

}  // namespace
