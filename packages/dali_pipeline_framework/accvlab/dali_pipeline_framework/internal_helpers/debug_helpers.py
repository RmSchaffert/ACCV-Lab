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

from functools import partial

from nvidia.dali import fn
import nvidia.dali.pipeline


def print_tensor_op(tensor_list: nvidia.dali.pipeline.DataNode, name: str):
    '''Python operator for printing a tensor list.

    This operator is used to print a tensor list while running the pipeline.
    Note that simply adding print statements to the pipeline will not work as
    those will be executed during the DALI graph construction phase, not the actual
    execution of the pipeline, and therefore, the printed objects will be
    :class:`DataNode` (i.e. "placeholder" objects) instead of the actual data.

    Note:
        The input is of type :class:`~nvidia.dali.pipeline.DataNode`, not a tensor list
        (:class:`~nvidia.dali.backend_impl.TensorListCPU` or
        :class:`~nvidia.dali.backend_impl.TensorListGPU`). This is because this function is called in the
        DALI pipeline definition, i.e. during the graph construction phase.

    Args:
        tensor_list: The tensor list to print.
        name: The name of the tensor list (will be shown in the printed output).
    '''

    def print_tensor_func(tensor_list_used, name_used):
        assert isinstance(
            tensor_list_used, list
        ), "Input has to be a tensor list. Use 'batch_processing=True' when using this function."
        size = len(tensor_list_used)
        print(f"-----{name_used}-----")
        print(f"size: {size}")
        for i in range(size):
            print(f"{name_used}[{i}]: ")
            print(tensor_list_used[i])
        print("-----")

    used_func = partial(print_tensor_func, name_used=name)
    fn.python_function(tensor_list, function=used_func, num_outputs=0, batch_processing=True, preserve=True)


def print_tensor_size_op(tensor_list: nvidia.dali.pipeline.DataNode, name: str):
    '''Python operator for printing the size of a tensor list.

    This operator is used to print the size of a tensor list while running the pipeline.
    See :func:`print_tensor_op` for more details on why this operator is needed.

    Note:
        The input is of type :class:`~nvidia.dali.pipeline.DataNode`, not a tensor list
        (:class:`~nvidia.dali.backend_impl.TensorListCPU` or
        :class:`~nvidia.dali.backend_impl.TensorListGPU`). This is because this function is called in the
        DALI pipeline definition, i.e. during the graph construction phase.

    Args:
        tensor_list: The tensor list to print the size of.
        name: The name of the tensor list (will be shown in the printed output).
    '''

    def print_tensor_func(tensor_list_used, name_used):
        assert isinstance(
            tensor_list_used, list
        ), "Input has to be a tensor list. Use 'batch_processing=True' when using this function."
        size = len(tensor_list_used)
        print(f"-----{name_used}-----")
        print(f"size: {size}")
        for i in range(size):
            print(f"{name_used}[{i}].shape: ")
            print(tensor_list_used[i].shape)
        print("-----")

    used_func = partial(print_tensor_func, name_used=name)
    fn.python_function(tensor_list, function=used_func, num_outputs=0, batch_processing=True, preserve=True)
