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

void indexing_forward_cuda(const torch::Tensor& input_data, const torch::Tensor& input_indices,
                           const torch::Tensor& input_nums_indices, torch::Tensor& result);

void indexing_backward_new_tensor_cuda(const torch::Tensor& grad, const torch::Tensor& input_indices,
                                       const torch::Tensor& input_nums_indices, torch::Tensor& result,
                                       double fill_value, bool backward_accumulate = true);

void indexing_backward_insert_cuda(const torch::Tensor& to_insert, const torch::Tensor& input_indices,
                                   const torch::Tensor& input_nums_indices, torch::Tensor& to_insert_into);

void indexing_backward_insert_const_cuda(double to_insert, const torch::Tensor& input_indices,
                                         const torch::Tensor& input_nums_indices,
                                         torch::Tensor& to_insert_into);

void map_values_by_index_pairs_cuda(const torch::Tensor& input_data, const torch::Tensor& input_indices,
                                    const torch::Tensor& output_indices, const torch::Tensor& nums_indices,
                                    torch::Tensor& to_insert_into, bool backward_accumulate = false);

void set_true_values_in_mask_cuda(const torch::Tensor& mask, const torch::Tensor& indices,
                                  const torch::Tensor& nums_indices, torch::Tensor& mask_to_set);

void set_ragged_batch_padded_to_filler_value_cuda(torch::Tensor& data,
                                                  const torch::Tensor& nums_valid_entries,
                                                  double filler_value);

static inline std::vector<int64_t> get_size_as_vec(const torch::Tensor& tensor) {
    const torch::IntArrayRef size = tensor.sizes();
    std::vector<int64_t> size_as_vec(size.begin(), size.end());
    return size_as_vec;
}

torch::Tensor indexing_forward(const torch::Tensor& input_data, const torch::Tensor& input_indices,
                               const torch::Tensor& input_nums_indices, double fill_value) {
    CHECK_CONTIGUOUS(input_data);
    CHECK_CONTIGUOUS(input_indices);
    CHECK_CONTIGUOUS(input_nums_indices);
    CHECK_SAME_CUDA_DEVICE(input_data, input_indices, input_nums_indices);

    CHECK_NUM_DIMS_AT_LEAST(input_nums_indices, 1);
    const size_t num_batch_dims = input_nums_indices.dim();

    CHECK_NUM_DIMS_AT_LEAST(input_indices, 1);
    CHECK_NUM_DIMS_AT_LEAST(input_data, num_batch_dims + 1);
    CHECK_SIZE_MATCH_FIRST_DIMS(input_data, input_indices, num_batch_dims);
    CHECK_SIZE_MATCH_FIRST_DIMS(input_indices, input_nums_indices, num_batch_dims);

    std::vector<int64_t> res_size(input_data.dim());
    for (size_t i = 0; i <= num_batch_dims; ++i) {
        res_size[i] = input_indices.size(i);
    }
    for (size_t i = num_batch_dims + 1; i < input_data.dim(); ++i) {
        res_size[i] = input_data.size(i);
    }

    torch::TensorOptions options = torch::TensorOptions()
                                       .dtype(input_data.scalar_type())
                                       .device(input_data.device())
                                       .requires_grad(input_data.requires_grad());

    torch::Tensor res = torch::full(res_size, fill_value, options);

    indexing_forward_cuda(input_data, input_indices, input_nums_indices, res);
    return res;
}

torch::Tensor indexing_backward_new_tensor(const torch::Tensor& to_insert, const torch::Tensor& input_indices,
                                           const torch::Tensor& input_nums_indices,
                                           const int64_t input_num_targets, double fill_value,
                                           bool backward_accumulate = true) {
    CHECK_CONTIGUOUS(to_insert);
    CHECK_CONTIGUOUS(input_indices);
    CHECK_CONTIGUOUS(input_nums_indices);
    CHECK_SAME_CUDA_DEVICE(to_insert, input_indices, input_nums_indices);

    CHECK_NUM_DIMS_AT_LEAST(input_nums_indices, 1);

    const size_t num_batch_dims = input_nums_indices.dim();

    CHECK_NUM_DIMS_AT_LEAST(to_insert, num_batch_dims + 1);
    CHECK_NUM_DIMS(input_indices, num_batch_dims + 1);
    CHECK_SIZE_MATCH_FIRST_DIMS(to_insert, input_indices, num_batch_dims);
    CHECK_SIZE_MATCH_FIRST_DIMS(input_indices, input_nums_indices, num_batch_dims);

    std::vector<int64_t> input_shape = get_size_as_vec(to_insert);
    input_shape[num_batch_dims] = input_num_targets;

    torch::TensorOptions options = torch::TensorOptions()
                                       .dtype(to_insert.scalar_type())
                                       .device(to_insert.device())
                                       .requires_grad(to_insert.requires_grad());

    torch::Tensor res = torch::full(input_shape, fill_value, options);

    indexing_backward_new_tensor_cuda(to_insert, input_indices, input_nums_indices, res, fill_value,
                                      backward_accumulate);

    return res;
}

torch::Tensor indexing_backward_insert(const torch::Tensor& to_insert, const torch::Tensor& input_indices,
                                       const torch::Tensor& input_nums_indices,
                                       const torch::Tensor& to_insert_into) {
    CHECK_CONTIGUOUS(to_insert);
    CHECK_CONTIGUOUS(input_indices);
    CHECK_CONTIGUOUS(input_nums_indices);
    CHECK_CONTIGUOUS(to_insert_into);
    CHECK_SAME_CUDA_DEVICE(to_insert, input_indices, input_nums_indices, to_insert_into);
    CHECK_SAME_DTYPE("Same dtype required for `to_insert` and `to_insert_into`", to_insert, to_insert_into);

    CHECK_NUM_DIMS_AT_LEAST(input_nums_indices, 1);

    const size_t num_batch_dims = input_nums_indices.dim();

    CHECK_NUM_DIMS_AT_LEAST(to_insert, num_batch_dims + 1);
    CHECK_NUM_DIMS(input_indices, num_batch_dims + 1);
    CHECK_SIZE_MATCH_FIRST_DIMS(to_insert, input_indices, num_batch_dims);
    CHECK_SIZE_MATCH_FIRST_DIMS(input_indices, input_nums_indices, num_batch_dims);
    CHECK_SIZE_MATCH_EXCEPT_DIM(to_insert, to_insert_into, num_batch_dims);

    torch::Tensor to_insert_into_clone = to_insert_into.clone();

    indexing_backward_insert_cuda(to_insert, input_indices, input_nums_indices, to_insert_into_clone);
    return to_insert_into_clone;
}

torch::Tensor indexing_backward_insert_const(double to_insert, const torch::Tensor& input_indices,
                                             const torch::Tensor& input_nums_indices,
                                             const torch::Tensor& to_insert_into) {
    CHECK_CONTIGUOUS(input_indices);
    CHECK_CONTIGUOUS(input_nums_indices);
    CHECK_CONTIGUOUS(to_insert_into);
    CHECK_SAME_CUDA_DEVICE(input_indices, input_nums_indices, to_insert_into);

    CHECK_NUM_DIMS_AT_LEAST(input_nums_indices, 1);

    const size_t num_batch_dims = input_nums_indices.dim();

    CHECK_NUM_DIMS_AT_LEAST(to_insert_into, num_batch_dims + 1);
    CHECK_NUM_DIMS(input_indices, num_batch_dims + 1);
    CHECK_SIZE_MATCH_FIRST_DIMS(input_indices, input_nums_indices, num_batch_dims);

    torch::Tensor to_insert_into_clone = to_insert_into.clone();

    indexing_backward_insert_const_cuda(to_insert, input_indices, input_nums_indices, to_insert_into_clone);
    return to_insert_into_clone;
}

torch::Tensor map_values_by_index_pairs(const torch::Tensor& input_data, const torch::Tensor& input_indices,
                                        const torch::Tensor& output_indices,
                                        const torch::Tensor& nums_indices,
                                        const torch::Tensor& to_insert_into,
                                        bool backward_accumulate = false) {
    CHECK_CONTIGUOUS(input_data);
    CHECK_CONTIGUOUS(input_indices);
    CHECK_CONTIGUOUS(output_indices);
    CHECK_CONTIGUOUS(nums_indices);
    CHECK_CONTIGUOUS(to_insert_into);
    CHECK_SAME_CUDA_DEVICE(input_data, input_indices, output_indices, nums_indices, to_insert_into);
    CHECK_SAME_DTYPE("Same dtype required for `input_data` and `to_insert_into`", input_data, to_insert_into);

    CHECK_NUM_DIMS_AT_LEAST(nums_indices, 1);

    const size_t num_batch_dims = nums_indices.dim();

    CHECK_NUM_DIMS_AT_LEAST(input_data, num_batch_dims + 1);
    CHECK_NUM_DIMS(input_indices, num_batch_dims + 1);
    CHECK_NUM_DIMS(output_indices, num_batch_dims + 1);
    CHECK_SIZE_MATCH_FIRST_DIMS(input_data, input_indices, num_batch_dims);
    CHECK_SIZE_MATCH(input_indices, output_indices);
    CHECK_SIZE_MATCH_FIRST_DIMS(input_indices, nums_indices, num_batch_dims);
    CHECK_SIZE_MATCH_EXCEPT_DIM(input_data, to_insert_into, num_batch_dims);

    torch::Tensor to_insert_into_clone = to_insert_into.clone();

    map_values_by_index_pairs_cuda(input_data, input_indices, output_indices, nums_indices,
                                   to_insert_into_clone, backward_accumulate);
    return to_insert_into_clone;
}

torch::Tensor get_mask_from_indices(const torch::Tensor& indices, const torch::Tensor& nums_indices,
                                    size_t num_targets) {
    CHECK_CONTIGUOUS(indices);
    CHECK_CONTIGUOUS(nums_indices);
    CHECK_SAME_CUDA_DEVICE(indices, nums_indices);

    CHECK_NUM_DIMS_AT_LEAST(nums_indices, 1);

    const size_t num_batch_dims = nums_indices.dim();

    CHECK_NUM_DIMS(indices, num_batch_dims + 1);
    CHECK_SIZE_MATCH_FIRST_DIMS(indices, nums_indices, num_batch_dims);

    torch::TensorOptions options =
        torch::TensorOptions().dtype(torch::kBool).device(indices.device()).requires_grad(false);

    std::vector<int64_t> res_size(num_batch_dims + 1);
    for (size_t i = 0; i < num_batch_dims; ++i) {
        res_size[i] = nums_indices.size(i);
    }
    res_size[num_batch_dims] = num_targets;

    torch::Tensor res = torch::zeros(res_size, options);

    set_true_values_in_mask_cuda(res, indices, nums_indices, res);
    return res;
}

void set_ragged_batch_padded_to_filler_value_in_place(torch::Tensor& data,
                                                      const torch::Tensor& nums_valid_entries,
                                                      double filler_value) {
    CHECK_CONTIGUOUS(data);
    CHECK_CONTIGUOUS(nums_valid_entries);
    CHECK_SAME_CUDA_DEVICE(data, nums_valid_entries);

    CHECK_NUM_DIMS_AT_LEAST(nums_valid_entries, 1);

    const size_t num_batch_dims = nums_valid_entries.dim();

    CHECK_NUM_DIMS_AT_LEAST(data, num_batch_dims + 1);
    CHECK_SIZE_MATCH_FIRST_DIMS(data, nums_valid_entries, num_batch_dims);

    set_ragged_batch_padded_to_filler_value_cuda(data, nums_valid_entries, filler_value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &indexing_forward, "Batched Indexing (CUDA)", py::arg("input_data"),
          py::arg("input_indices"), py::arg("input_nums_indices"), py::arg("fill_value") = 0.0);
    m.def("backward_new_tensor", &indexing_backward_new_tensor, "", py::arg("to_insert"),
          py::arg("input_indices"), py::arg("input_nums_indices"), py::arg("input_num_targets"),
          py::arg("fill_value") = 0.0, py::arg("backward_accumulate") = true);
    m.def("backward_insert", &indexing_backward_insert, "", py::arg("to_insert"), py::arg("input_indices"),
          py::arg("input_nums_indices"), py::arg("to_insert_into"));
    m.def("backward_insert_const", &indexing_backward_insert_const, "", py::arg("to_insert"),
          py::arg("input_indices"), py::arg("input_nums_indices"), py::arg("to_insert_into"));
    m.def("map_values_by_index_pairs", &map_values_by_index_pairs, "", py::arg("input_data"),
          py::arg("input_indices"), py::arg("output_indices"), py::arg("nums_indices"),
          py::arg("to_insert_into"), py::arg("backward_accumulate") = false);
    m.def("get_mask_from_indices", &get_mask_from_indices, "", py::arg("indices"), py::arg("nums_indices"),
          py::arg("num_targets"));
    m.def("set_ragged_batch_padded_to_filler_value_in_place",
          &set_ragged_batch_padded_to_filler_value_in_place, "", py::arg("data"),
          py::arg("nums_valid_entries"), py::arg("filler_value"));
}
