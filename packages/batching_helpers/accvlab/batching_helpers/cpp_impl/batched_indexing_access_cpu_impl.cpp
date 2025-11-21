/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>

#include <torch/torch.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <torch/extension.h>

#include "batched_indexing_access_helpers.h"

template <typename dtype, typename index_type>
void set_ragged_batch_padded_to_filler_value_cpu_impl(dtype* data, const index_type* nums_valid_entries,
                                                      int64_t batch_size, int64_t max_sample_size,
                                                      int64_t num_data_elements_per_index,
                                                      dtype filler_value) {
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            const index_type num_valid_entries = nums_valid_entries[i];
            const size_t row_idx = i * max_sample_size;
            for (index_type j = num_valid_entries; j < max_sample_size; ++j) {
                const size_t entry_idx = (row_idx + j) * num_data_elements_per_index;
                for (int64_t k = 0; k < num_data_elements_per_index; ++k) {
                    data[entry_idx + k] = filler_value;
                }
            }
        }
    });
}

void set_ragged_batch_padded_to_filler_value_cpu(torch::Tensor& data, const torch::Tensor& nums_valid_entries,
                                                 double filler_value) {
    const int64_t num_batch_dims = nums_valid_entries.dim();
    const int64_t batch_numel = nums_valid_entries.numel();

    const int64_t num_data_elements_per_index = get_number_data_elements_per_index(data, num_batch_dims + 1);
    const int64_t max_sample_size = data.size(num_batch_dims);

    DISPATCH_INDEX_TYPES(
        nums_valid_entries.scalar_type(),
        "set_ragged_batch_padded_to_filler_value_cpu [for: nums_valid_entries]", [&] {
            using index_scalar_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND5(
                at::ScalarType::Long, at::ScalarType::Int, at::ScalarType::Half, at::ScalarType::BFloat16,
                at::ScalarType::Bool, data.scalar_type(),
                "set_ragged_batch_padded_to_filler_value_cpu [for: data]", [&] {
                    set_ragged_batch_padded_to_filler_value_cpu_impl<scalar_t, index_scalar_t>(
                        data.data_ptr<scalar_t>(), nums_valid_entries.data_ptr<index_scalar_t>(), batch_numel,
                        max_sample_size, num_data_elements_per_index, filler_value);
                });
        });
}