# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import accvlab.batching_helpers.batched_indexing_access_cuda as batched_indexing_access_cuda
import accvlab.batching_helpers.batched_indexing_access_cpu as batched_indexing_access_cpu


class SetPaddedTo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, sample_sizes, value_to_set):
        ctx.save_for_backward(sample_sizes)
        data = data.contiguous()
        sample_sizes = sample_sizes.to(dtype=torch.int64).contiguous()
        if data.device.type == "cuda":
            batched_indexing_access_cuda.set_ragged_batch_padded_to_filler_value_in_place(
                data, sample_sizes, value_to_set
            )
        else:
            batched_indexing_access_cpu.set_ragged_batch_padded_to_filler_value_in_place(
                data, sample_sizes, value_to_set
            )
        return data

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None
        else:
            grad_input = grad_output.clone()
            SetPaddedTo.apply(grad_input, ctx.saved_tensors[0], 0.0)
            return grad_input, None, None
