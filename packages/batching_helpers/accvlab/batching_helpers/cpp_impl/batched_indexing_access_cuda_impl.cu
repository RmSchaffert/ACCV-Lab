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

#include <cuda.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "batched_indexing_access_helpers.h"

__device__ __forceinline__ void lock_mutex(int32_t* mutex_ptr) {
    while (atomicCAS(mutex_ptr, static_cast<int32_t>(0), static_cast<int32_t>(1)) != static_cast<int32_t>(0))
        ;
}

__device__ __forceinline__ void unlock_mutex(int32_t* mutex_ptr) {
    atomicExch(mutex_ptr, static_cast<int32_t>(0));
}

template <typename T>
__device__ __forceinline__ void atomicSetFirstThenAdd(T* data_ptr, bool* is_not_first_mask_ptr,
                                                      int32_t* mutex_ptr, T val) {
    lock_mutex(mutex_ptr);
    if (*is_not_first_mask_ptr) {
        *data_ptr += val;
    } else {
        *data_ptr = val;
        *is_not_first_mask_ptr = true;
    }
    unlock_mutex(mutex_ptr);
}

template <typename dtype, typename index_type, typename nums_indices_type>
__global__ static void indexing_kernel(dtype* unindexed_data, const index_type* indices,
                                       const nums_indices_type* nums_indices, size_t height,
                                       size_t width_input, size_t width_output,
                                       size_t num_data_elements_per_index, dtype* data_at_indices,
                                       bool is_forward_direction, bool* backward_touched_mask,
                                       int32_t* mutexes, bool backward_accumulate = true) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    size_t j_out = blockDim.y * blockIdx.y + threadIdx.y;
    size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i < height && j_out < width_output && k < num_data_elements_per_index) {
        // Get the number of valid indices for the current row (sample)
        const nums_indices_type num_elements_in_row = nums_indices[i];
        // If we are inside the valid indices
        if (j_out < num_elements_in_row) {
            // Compute the index in the index tensor which is processed by the current thread
            const size_t idx_idx = i * width_output + j_out;
            // Index inside the indexed data (output for the forward direction) which is processed by the current thread
            const size_t idx_data_at_indices = idx_idx * num_data_elements_per_index + k;
            // Get the index from the index tensor which is processed by the current thread
            index_type idx_j = indices[idx_idx];
            // If index is negative, convert to corresponding positive index
            if (idx_j < 0) {
                idx_j = width_input + idx_j;
            }
            // Make sure we are not out of bounds
            CUDA_KERNEL_ASSERT(idx_j >= 0 && idx_j < width_input && "Index out of bounds");
            // Get the entry index in the input, considering only the i- and j-dimensions for now
            const size_t idx_unindexted_data_2d_ij = i * width_input + static_cast<size_t>(idx_j);
            // Get the element index in the input, also considering that each entry has multiple elements
            const size_t idx_unindexted_data = idx_unindexted_data_2d_ij * num_data_elements_per_index + k;

            // If the computation is in the forward direction
            if (is_forward_direction) {
                // write the input to the corresponding output
                data_at_indices[idx_data_at_indices] = unindexed_data[idx_unindexted_data];
            }
            // Otherwise (i.e. backward direction)
            else {
                // Write the output to the corresponding input. There are two modes for this operation:
                // 1. If the values are not accumulated:
                //    In case multiple indices in one row point to the same `unindexed_data` element, one
                //    of the corresponding `data_at_indices` values is picked arbitrarily and written to `unindexed_data`.
                // 2. If the values are accumulated:
                //    In case multiple indices in one row point to the same `unindexed_data` element, the values are accumulated.
                //    Note that in this case, the first value written must replace the orinal value, while the remaining
                //    writes must instead add to the value.
                //    Note that this is the mode used in the backward pass of `batched_index_access()`.
                if (!backward_accumulate) {
                    // Case 1.
                    unindexed_data[idx_unindexted_data] = data_at_indices[idx_data_at_indices];
                } else {
                    // Case 2.
                    atomicSetFirstThenAdd(
                        unindexed_data + idx_unindexted_data, backward_touched_mask + idx_unindexted_data,
                        mutexes + idx_unindexted_data, data_at_indices[idx_data_at_indices]);
                }
            }
        }
    }
}

template <typename dtype, typename index_type>
__global__ static void map_values_by_index_pairs_kernel(
    const dtype* input_data, const index_type* input_indices, const index_type* output_indices,
    const index_type* nums_indices, size_t width_input, size_t width_indices, size_t width_output,
    size_t height, size_t num_data_elements_per_index, dtype* data_to_write_in,
    bool* backward_touched_mask = nullptr, int32_t* mutexes = nullptr, bool backward_accumulate = false) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    size_t j_index = blockDim.y * blockIdx.y + threadIdx.y;
    size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i < height && j_index < width_indices && k < num_data_elements_per_index) {
        // Get the number of valid indices for the current row (sample)
        const index_type num_elements_in_row = nums_indices[i];
        // If we are not outside the valid indices
        if (j_index < num_elements_in_row) {
            // Compute the index in the index tensor which is processed by the current thread
            const size_t idx_idx = i * width_indices + j_index;
            // Get the indices from the index tensors which are processed by the current thread
            index_type idx_in = input_indices[idx_idx];
            index_type idx_out = output_indices[idx_idx];
            // If indices are negative, convert to corresponding positive indices
            if (idx_in < 0) {
                idx_in = width_input + idx_in;
            }
            if (idx_out < 0) {
                idx_out = width_output + idx_out;
            }
            // Make sure we are not out of bounds
            CUDA_KERNEL_ASSERT(idx_in >= 0 && idx_in < width_input && idx_out >= 0 &&
                               idx_out < width_output && "Index out of bounds");
            // Get the entry indices, considering only the i- and j-dimensions for now
            const size_t idx_input_2d_ij = i * width_input + static_cast<size_t>(idx_in);
            const size_t idx_output_2d_ij = i * width_output + static_cast<size_t>(idx_out);
            // Get the element indices
            const size_t idx_input = idx_input_2d_ij * num_data_elements_per_index + k;
            const size_t idx_output = idx_output_2d_ij * num_data_elements_per_index + k;

            if (!backward_accumulate) {
                data_to_write_in[idx_output] = input_data[idx_input];
            } else {
                atomicSetFirstThenAdd(data_to_write_in + idx_output, backward_touched_mask + idx_output,
                                      mutexes + idx_output, input_data[idx_input]);
            }
        }
    }
}

template <typename dtype, typename index_type, typename nums_indices_type>
__global__ static void insert_const_at_indices_kernel(dtype const_val, const index_type* input_indices,
                                                      const nums_indices_type* input_nums_indices,
                                                      size_t height, size_t width_indices,
                                                      size_t width_output, size_t num_data_elements_per_index,
                                                      dtype* data_to_insert_in) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    size_t j_idx = blockDim.y * blockIdx.y + threadIdx.y;
    size_t k = blockDim.z * blockIdx.z + threadIdx.z;
    if (i < height && j_idx < width_indices && k < num_data_elements_per_index) {
        // Get the number of indices in the current row (sample)
        const nums_indices_type num_elements_in_row = input_nums_indices[i];
        // If we are not outside the valid indices
        if (j_idx < num_elements_in_row) {
            // Get the index for the correct element in the index tensor
            const size_t idx_idx = i * width_indices + j_idx;
            // Get the index at which to write the conatant value from the index tensor
            index_type j_out = input_indices[idx_idx];
            // If index is negative, compute the corresponding positive index
            if (j_out < 0) {
                j_out = width_output + j_out;
            }
            // Make sure are not out of bounds
            CUDA_KERNEL_ASSERT(j_out >= 0 && j_out < width_output && "Index out of bounds");
            // Index of the entry to write constant to
            const size_t entry_idx = i * width_output + j_out;
            // Index of the element corresponding to the current kernel (every entry potentially containing multiple elements)
            const size_t elem_idx = entry_idx * num_data_elements_per_index + k;
            // Write the constant
            data_to_insert_in[elem_idx] = const_val;
        }
    }
}

template <typename dtype, typename index_type>
__global__ static void set_ragged_batch_padded_to_filler_value_kernel(dtype* data,
                                                                      const index_type* nums_valid_entries,
                                                                      size_t height, size_t width,
                                                                      size_t num_data_elements_per_index,
                                                                      dtype filler_value) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i < height && j < width && k < num_data_elements_per_index) {
        const index_type num_valid_entries = nums_valid_entries[i];
        if (j >= num_valid_entries) {
            const size_t idx_data = (i * width + j) * num_data_elements_per_index + k;
            data[idx_data] = filler_value;
        }
    }
}

static size_t ceil_pow2(size_t value) {
    size_t count = 0;
    if (value > 0) {
        value -= 1;
        while (value != 0) {
            value >>= 1;
            ++count;
        }
    }
    const size_t res = 1 << count;
    return res;
}

static void setup_grid(size_t height, size_t width, size_t num_data_elements_per_index, dim3& grid_size,
                       dim3& block_size) {
    size_t left_block_numel = 1024;

    size_t block_size_z = ceil_pow2(static_cast<size_t>(num_data_elements_per_index));
    block_size_z = std::min(block_size_z, static_cast<size_t>(32));
    left_block_numel /= block_size_z;

    size_t block_size_y = ceil_pow2(static_cast<size_t>(width));
    const size_t max_block_size_y = std::min(left_block_numel, static_cast<size_t>(32));
    block_size_y = std::min(block_size_y, max_block_size_y);
    left_block_numel /= block_size_y;

    size_t block_size_x = ceil_pow2(static_cast<size_t>(height));
    block_size_x = std::min(block_size_x, left_block_numel);

    block_size = {static_cast<unsigned int>(block_size_x), static_cast<unsigned int>(block_size_y),
                  static_cast<unsigned int>(block_size_z)};
    grid_size.x = (height + block_size_x - 1) / block_size_x;
    grid_size.y = (width + block_size_y - 1) / block_size_y;
    grid_size.z = (num_data_elements_per_index + block_size_z - 1) / block_size_z;
}

void indexing_forward_cuda(const torch::Tensor& input_data, const torch::Tensor& input_indices,
                           const torch::Tensor& input_nums_indices, torch::Tensor& result) {
    if (input_indices.numel() == 0) {
        return;
    }

    const int64_t num_batch_dims = input_nums_indices.dim();
    const int64_t batch_numel = input_nums_indices.numel();
    const int64_t num_data_elements_per_index =
        get_number_data_elements_per_index(input_data, num_batch_dims + 1);

    dim3 grid_size;
    dim3 block_size;
    setup_grid(batch_numel, input_indices.size(num_batch_dims), num_data_elements_per_index, grid_size,
               block_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_INDEX_TYPES(input_indices.scalar_type(), "indexing_forward_cuda [for: input_indices]", [&] {
        using indices_scalar_t = scalar_t;
        DISPATCH_INDEX_TYPES(
            input_nums_indices.scalar_type(), "indexing_forward_cuda [for: input_nums_indices]", [&] {
                using nums_indices_scalar_t = scalar_t;
                AT_DISPATCH_FLOATING_TYPES_AND4(
                    at::ScalarType::Long, at::ScalarType::Int, at::ScalarType::Half, at::ScalarType::BFloat16,
                    input_data.scalar_type(), "indexing_forward_cuda [for: data]", [&] {
                        indexing_kernel<<<grid_size, block_size, 0, stream>>>(
                            input_data.data_ptr<scalar_t>(), input_indices.data_ptr<indices_scalar_t>(),
                            input_nums_indices.data_ptr<nums_indices_scalar_t>(), batch_numel,
                            input_data.size(num_batch_dims), input_indices.size(num_batch_dims),
                            num_data_elements_per_index, result.data_ptr<scalar_t>(), true, nullptr, nullptr,
                            false);
                        C10_CUDA_CHECK(cudaGetLastError());
                    });
            });
    });
}

void indexing_backward_new_tensor_cuda(const torch::Tensor& grad, const torch::Tensor& input_indices,
                                       const torch::Tensor& input_nums_indices, torch::Tensor& result,
                                       double fill_value, bool backward_accumulate = true) {
    if (input_indices.numel() == 0) {
        return;
    }

    const int64_t num_batch_dims = input_nums_indices.dim();
    const int64_t batch_numel = input_nums_indices.numel();
    const int64_t num_data_elements_per_index = get_number_data_elements_per_index(grad, num_batch_dims + 1);

    torch::Tensor touched_mask;
    torch::Tensor mutexes;
    if (backward_accumulate) {
        torch::TensorOptions mask_options =
            torch::TensorOptions().dtype(at::kBool).device(grad.device()).requires_grad(false);
        touched_mask = torch::zeros(result.numel(), mask_options);
        torch::TensorOptions mutex_options =
            torch::TensorOptions().dtype(at::ScalarType::Int).device(grad.device()).requires_grad(false);
        mutexes = torch::zeros(result.numel(), mutex_options);
    }

    dim3 grid_size;
    dim3 block_size;
    setup_grid(batch_numel, input_indices.size(num_batch_dims), num_data_elements_per_index, grid_size,
               block_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    bool* touched_mask_data = backward_accumulate ? touched_mask.data_ptr<bool>() : nullptr;
    int32_t* mutexes_data = backward_accumulate ? mutexes.data_ptr<int32_t>() : nullptr;

    DISPATCH_INDEX_TYPES(
        input_indices.scalar_type(), "indexing_backward_new_tensor_cuda [for: input_indices]", [&] {
            using indices_scalar_t = scalar_t;
            DISPATCH_INDEX_TYPES(
                input_nums_indices.scalar_type(),
                "indexing_backward_new_tensor_cuda [for: input_nums_indices]", [&] {
                    using nums_indices_scalar_t = scalar_t;
                    AT_DISPATCH_FLOATING_TYPES_AND4(
                        at::ScalarType::Long, at::ScalarType::Int, at::ScalarType::Half,
                        at::ScalarType::BFloat16, grad.scalar_type(),
                        "indexing_backward_new_tensor_cuda [for: data]", [&] {
                            indexing_kernel<<<grid_size, block_size, 0, stream>>>(
                                result.data_ptr<scalar_t>(), input_indices.data_ptr<indices_scalar_t>(),
                                input_nums_indices.data_ptr<nums_indices_scalar_t>(), batch_numel,
                                result.size(num_batch_dims), input_indices.size(num_batch_dims),
                                num_data_elements_per_index, grad.data_ptr<scalar_t>(), false,
                                touched_mask_data, mutexes_data, backward_accumulate);
                            C10_CUDA_CHECK(cudaGetLastError());
                        });
                });
        });
}

void indexing_backward_insert_cuda(const torch::Tensor& to_insert, const torch::Tensor& input_indices,
                                   const torch::Tensor& input_nums_indices, torch::Tensor& to_insert_into) {
    if (input_indices.numel() == 0) {
        return;
    }

    const int64_t num_batch_dims = input_nums_indices.dim();
    const int64_t batch_numel = input_nums_indices.numel();
    const int64_t num_data_elements_per_index =
        get_number_data_elements_per_index(to_insert, num_batch_dims + 1);

    dim3 grid_size;
    dim3 block_size;
    setup_grid(batch_numel, input_indices.size(num_batch_dims), num_data_elements_per_index, grid_size,
               block_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_INDEX_TYPES(
        input_indices.scalar_type(), "indexing_backward_insert_cuda [for: input_indices]", [&] {
            using indices_scalar_t = scalar_t;
            DISPATCH_INDEX_TYPES(input_nums_indices.scalar_type(),
                                 "indexing_backward_insert_cuda [for: input_nums_indices]", [&] {
                                     using nums_indices_scalar_t = scalar_t;
                                     AT_DISPATCH_FLOATING_TYPES_AND4(
                                         at::ScalarType::Long, at::ScalarType::Int, at::ScalarType::Half,
                                         at::ScalarType::BFloat16, to_insert.scalar_type(),
                                         "indexing_backward_insert_cuda [for: data]", [&] {
                                             indexing_kernel<<<grid_size, block_size, 0, stream>>>(
                                                 to_insert_into.data_ptr<scalar_t>(),
                                                 input_indices.data_ptr<indices_scalar_t>(),
                                                 input_nums_indices.data_ptr<nums_indices_scalar_t>(),
                                                 batch_numel, to_insert_into.size(num_batch_dims),
                                                 input_indices.size(num_batch_dims),
                                                 num_data_elements_per_index, to_insert.data_ptr<scalar_t>(),
                                                 false, nullptr, nullptr, false);
                                             C10_CUDA_CHECK(cudaGetLastError());
                                         });
                                 });
        });
}

void indexing_backward_insert_const_cuda(double to_insert, const torch::Tensor& input_indices,
                                         const torch::Tensor& input_nums_indices,
                                         torch::Tensor& to_insert_into) {
    if (input_indices.numel() == 0) {
        return;
    }

    const int64_t num_batch_dims = input_nums_indices.dim();
    const int64_t batch_numel = input_nums_indices.numel();
    const int64_t num_data_elements_per_index =
        get_number_data_elements_per_index(to_insert_into, num_batch_dims + 1);

    dim3 grid_size;
    dim3 block_size;
    setup_grid(batch_numel, input_indices.size(num_batch_dims), num_data_elements_per_index, grid_size,
               block_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_INDEX_TYPES(
        input_indices.scalar_type(), "indexing_backward_insert_const_cuda [for: input_indices]", [&] {
            using indices_scalar_t = scalar_t;
            DISPATCH_INDEX_TYPES(
                input_nums_indices.scalar_type(),
                "indexing_backward_insert_const_cuda [for: input_nums_indices]", [&] {
                    using nums_indices_scalar_t = scalar_t;
                    AT_DISPATCH_FLOATING_TYPES_AND4(
                        at::ScalarType::Long, at::ScalarType::Int, at::ScalarType::Half,
                        at::ScalarType::BFloat16, to_insert_into.scalar_type(),
                        "indexing_backward_insert_const_cuda [for: data]", [&] {
                            insert_const_at_indices_kernel<<<grid_size, block_size, 0, stream>>>(
                                static_cast<scalar_t>(to_insert), input_indices.data_ptr<indices_scalar_t>(),
                                input_nums_indices.data_ptr<nums_indices_scalar_t>(), batch_numel,
                                input_indices.size(num_batch_dims), to_insert_into.size(num_batch_dims),
                                num_data_elements_per_index, to_insert_into.data_ptr<scalar_t>());
                            C10_CUDA_CHECK(cudaGetLastError());
                        });
                });
        });
}

void map_values_by_index_pairs_cuda(const torch::Tensor& input_data, const torch::Tensor& input_indices,
                                    const torch::Tensor& output_indices, const torch::Tensor& nums_indices,
                                    torch::Tensor& to_insert_into, bool backward_accumulate = false) {
    if (input_indices.numel() == 0) {
        return;
    }

    const int64_t num_batch_dims = nums_indices.dim();
    const int64_t batch_numel = nums_indices.numel();
    const int64_t num_data_elements_per_index =
        get_number_data_elements_per_index(to_insert_into, num_batch_dims + 1);

    torch::Tensor touched_mask;
    torch::Tensor mutexes;
    if (backward_accumulate) {
        torch::TensorOptions mask_options =
            torch::TensorOptions().dtype(at::kBool).device(to_insert_into.device()).requires_grad(false);
        touched_mask = torch::zeros(to_insert_into.numel(), mask_options);
        torch::TensorOptions mutex_options = torch::TensorOptions()
                                                 .dtype(at::ScalarType::Int)
                                                 .device(to_insert_into.device())
                                                 .requires_grad(false);
        mutexes = torch::zeros(to_insert_into.numel(), mutex_options);
    }

    dim3 grid_size;
    dim3 block_size;
    setup_grid(batch_numel, input_indices.size(num_batch_dims), num_data_elements_per_index, grid_size,
               block_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    bool* touched_mask_data = backward_accumulate ? touched_mask.data_ptr<bool>() : nullptr;
    int32_t* mutexes_data = backward_accumulate ? mutexes.data_ptr<int32_t>() : nullptr;

    DISPATCH_INDEX_TYPES(
        input_indices.scalar_type(), "map_values_by_index_pairs_cuda [for: indices & sizes]", [&] {
            using indices_scalar_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND4(
                at::ScalarType::Long, at::ScalarType::Int, at::ScalarType::Half, at::ScalarType::BFloat16,
                to_insert_into.scalar_type(), "map_values_by_index_pairs_cuda", [&] {
                    map_values_by_index_pairs_kernel<<<grid_size, block_size, 0, stream>>>(
                        input_data.data_ptr<scalar_t>(), input_indices.data_ptr<indices_scalar_t>(),
                        output_indices.data_ptr<indices_scalar_t>(),
                        nums_indices.data_ptr<indices_scalar_t>(), input_data.size(num_batch_dims),
                        input_indices.size(num_batch_dims), to_insert_into.size(num_batch_dims), batch_numel,
                        num_data_elements_per_index, to_insert_into.data_ptr<scalar_t>(), touched_mask_data,
                        mutexes_data, backward_accumulate);
                    C10_CUDA_CHECK(cudaGetLastError());
                });
        });
}

void set_true_values_in_mask_cuda(const torch::Tensor& mask, const torch::Tensor& indices,
                                  const torch::Tensor& nums_indices, torch::Tensor& mask_to_set) {
    if (indices.numel() == 0) {
        return;
    }

    const int64_t num_batch_dims = nums_indices.dim();
    const int64_t batch_numel = nums_indices.numel();

    dim3 grid_size;
    dim3 block_size;
    setup_grid(batch_numel, indices.size(num_batch_dims), 1, grid_size, block_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_INDEX_TYPES(indices.scalar_type(), "set_true_values_in_mask_cuda [for: indices]", [&] {
        using indices_scalar_t = scalar_t;
        DISPATCH_INDEX_TYPES(
            nums_indices.scalar_type(), "set_true_values_in_mask_cuda [for: nums_indices]", [&] {
                using nums_indices_scalar_t = scalar_t;
                // Note that in this case, only index (and size) tensors are used, and therefore there
                // is no need to dispatch over the data type (apart from the index type).
                insert_const_at_indices_kernel<<<grid_size, block_size, 0, stream>>>(
                    true, indices.data_ptr<indices_scalar_t>(),
                    nums_indices.data_ptr<nums_indices_scalar_t>(), batch_numel, indices.size(num_batch_dims),
                    mask_to_set.size(num_batch_dims), 1, mask_to_set.data_ptr<bool>());
                C10_CUDA_CHECK(cudaGetLastError());
            });
    });
}

void set_ragged_batch_padded_to_filler_value_cuda(torch::Tensor& data,
                                                  const torch::Tensor& nums_valid_entries,
                                                  double filler_value) {
    if (data.numel() == 0) {
        return;
    }

    const int64_t num_batch_dims = nums_valid_entries.dim();
    const int64_t batch_numel = nums_valid_entries.numel();
    const int64_t num_data_elements_per_index = get_number_data_elements_per_index(data, num_batch_dims + 1);

    dim3 grid_size;
    dim3 block_size;
    setup_grid(batch_numel, data.size(num_batch_dims), num_data_elements_per_index, grid_size, block_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_INDEX_TYPES(
        nums_valid_entries.scalar_type(),
        "set_ragged_batch_padded_to_filler_value_cuda [for: nums_valid_entries]", [&] {
            using index_scalar_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND5(
                at::ScalarType::Long, at::ScalarType::Int, at::ScalarType::Half, at::ScalarType::BFloat16,
                at::ScalarType::Bool, data.scalar_type(),
                "set_ragged_batch_padded_to_filler_value_cuda [for: data]", [&] {
                    set_ragged_batch_padded_to_filler_value_kernel<<<grid_size, block_size, 0, stream>>>(
                        data.data_ptr<scalar_t>(), nums_valid_entries.data_ptr<index_scalar_t>(), batch_numel,
                        data.size(num_batch_dims), num_data_elements_per_index,
                        static_cast<scalar_t>(filler_value));
                    C10_CUDA_CHECK(cudaGetLastError());
                });
        });
}
