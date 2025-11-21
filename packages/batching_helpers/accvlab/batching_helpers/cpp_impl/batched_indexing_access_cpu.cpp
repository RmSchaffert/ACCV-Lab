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

#include "batched_indexing_access_helpers.h"

void set_ragged_batch_padded_to_filler_value_cpu(torch::Tensor& data, const torch::Tensor& nums_valid_entries,
                                                 double filler_value);

void set_ragged_batch_padded_to_filler_value_in_place(torch::Tensor& data,
                                                      const torch::Tensor& nums_valid_entries,
                                                      double filler_value) {
    CHECK_CONTIGUOUS(data);
    CHECK_CONTIGUOUS(nums_valid_entries);
    CHECK_CPU(data);
    CHECK_CPU(nums_valid_entries);

    const int64_t num_batch_dims = nums_valid_entries.dim();

    CHECK_NUM_DIMS_AT_LEAST(data, num_batch_dims + 1);

    CHECK_SIZE_MATCH_FIRST_DIMS(data, nums_valid_entries, num_batch_dims);

    set_ragged_batch_padded_to_filler_value_cpu(data, nums_valid_entries, filler_value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_ragged_batch_padded_to_filler_value_in_place",
          &set_ragged_batch_padded_to_filler_value_in_place, "", py::arg("data"),
          py::arg("nums_valid_entries"), py::arg("filler_value"));
}