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

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

using accvlab::multi_tensor_copier::internal::H2DTransferChunk;
using accvlab::multi_tensor_copier::internal::H2DTransferSubmitter;
using accvlab::multi_tensor_copier::internal::plan_h2d_transfer_chunks;

class H2DTransferSubmitterTest : public ::testing::Test {
   protected:
    void SetUp() override {
        int device_count = 0;
        const auto status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device required";
        }
        device_ = c10::Device(c10::kCUDA, 0);
        stream_ = at::cuda::getStreamFromPool(false, 0);
    }

    at::Tensor pinned_arange(int64_t count, at::ScalarType dtype = at::kFloat) const {
        return at::arange(count, at::TensorOptions().dtype(dtype).device(at::kCPU)).pin_memory();
    }

    c10::Device device_{c10::kCUDA, 0};
    at::cuda::CUDAStream stream_ = at::cuda::getDefaultCUDAStream();
};

TEST_F(H2DTransferSubmitterTest, DisablesChunkingForZeroLimit) {
    const auto source = pinned_arange(10);
    const auto destination = at::empty(source.sizes(), source.options().device(device_));

    const auto chunks = plan_h2d_transfer_chunks(destination, source, 0);

    EXPECT_TRUE(chunks.empty());
}

TEST_F(H2DTransferSubmitterTest, PlansElementAlignedChunksAndRemainder) {
    const auto source = pinned_arange(10);
    const auto destination = at::empty(source.sizes(), source.options().device(device_));

    const auto chunks = plan_h2d_transfer_chunks(destination, source, 12);

    const std::vector<H2DTransferChunk> expected = {
        {0, 3},
        {3, 3},
        {6, 3},
        {9, 1},
    };
    ASSERT_EQ(chunks.size(), expected.size());
    for (size_t index = 0; index < chunks.size(); ++index) {
        EXPECT_EQ(chunks[index].element_offset, expected[index].element_offset);
        EXPECT_EQ(chunks[index].element_count, expected[index].element_count);
    }
}

TEST_F(H2DTransferSubmitterTest, UsesOneElementWhenByteLimitIsSmallerThanElement) {
    const auto source = pinned_arange(4, at::kDouble);
    const auto destination = at::empty(source.sizes(), source.options().device(device_));

    const auto chunks = plan_h2d_transfer_chunks(destination, source, 1);

    ASSERT_EQ(chunks.size(), 4);
    for (size_t index = 0; index < chunks.size(); ++index) {
        EXPECT_EQ(chunks[index].element_offset, static_cast<int64_t>(index));
        EXPECT_EQ(chunks[index].element_count, 1);
    }
}

TEST_F(H2DTransferSubmitterTest, FallsBackForUnsupportedTransfers) {
    const auto contiguous_source = pinned_arange(12);
    const auto non_contiguous_source = contiguous_source.reshape({3, 4}).transpose(0, 1);
    const auto cuda_destination =
        at::empty(contiguous_source.sizes(), contiguous_source.options().device(device_));
    const auto cpu_destination = at::empty_like(contiguous_source);
    const auto mismatched_dtype =
        at::empty(contiguous_source.sizes(), contiguous_source.options().dtype(at::kDouble).device(device_));
    const auto mismatched_size =
        at::empty({contiguous_source.numel() + 1}, contiguous_source.options().device(device_));

    EXPECT_TRUE(plan_h2d_transfer_chunks(cuda_destination, non_contiguous_source, 16).empty());
    EXPECT_TRUE(plan_h2d_transfer_chunks(cpu_destination, contiguous_source, 16).empty());
    EXPECT_TRUE(plan_h2d_transfer_chunks(mismatched_dtype, contiguous_source, 16).empty());
    EXPECT_TRUE(plan_h2d_transfer_chunks(mismatched_size, contiguous_source, 16).empty());
}

TEST_F(H2DTransferSubmitterTest, CopiesChunksOnNonDefaultStream) {
    const auto source = pinned_arange(4097);
    auto destination = at::empty(source.sizes(), source.options().device(device_));
    H2DTransferSubmitter submitter(stream_, 4096);

    {
        c10::cuda::CUDAGuard guard(stream_.device_index());
        at::cuda::CUDAStreamGuard stream_guard(stream_);
        submitter.submit(destination, source, true);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream_.stream()), cudaSuccess);

    EXPECT_TRUE(at::equal(destination.cpu(), source));
}

TEST_F(H2DTransferSubmitterTest, PreservesMultipleRequestsThroughOneSubmitter) {
    const auto first_source = pinned_arange(1025);
    const auto second_source = pinned_arange(2049, at::kLong);
    auto first_destination = at::empty(first_source.sizes(), first_source.options().device(device_));
    auto second_destination = at::empty(second_source.sizes(), second_source.options().device(device_));
    H2DTransferSubmitter submitter(stream_, 1024);

    {
        c10::cuda::CUDAGuard guard(stream_.device_index());
        at::cuda::CUDAStreamGuard stream_guard(stream_);
        submitter.submit(first_destination, first_source, true);
        submitter.submit(second_destination, second_source, true);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream_.stream()), cudaSuccess);

    EXPECT_TRUE(at::equal(first_destination.cpu(), first_source));
    EXPECT_TRUE(at::equal(second_destination.cpu(), second_source));
}

}  // namespace
