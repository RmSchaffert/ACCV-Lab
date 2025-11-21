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

import numpy as np
import cupy as cp
import nvidia.dali.pipeline
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import do_not_convert


@do_not_convert
def check_type(input: nvidia.dali.pipeline.DataNode, expected_type_np: np.dtype, identifier: str):
    '''Python operator for checking the type of the input.

    If the type of the input does not match the expected type, a ValueError is raised.

    Note:
        The check is performed at runtime, i.e. not the graph construction time.

    Important:
        Use the output of this function to tie the dependency on the check op, preventing it from being
        pruned or remain unused. Note that even though ``preserve=True`` is set in the operator performing the
        check, the operator may be not executed if the output is not used, meaning that no check will be
        performed.

    Args:
        input: The input to check the type of.
        expected_type_np: The expected type of the input.
        identifier: The identifier of the input.

    Returns:
        The same input. Use this output to tie the dependency on the check op, preventing it from being
        not called due to the unused output.

    Raises:
        ValueError: If the type of the input does not match the expected type.
    '''

    def check_type_fn(input):
        # Perform the check and raise an error if the type does not match.
        if input.dtype != expected_type_np:
            raise ValueError(
                f"Expected type ({expected_type_np}) and actual data type ({input.dtype}) do not match for '{identifier}'"
            )
        # Return the shape of the input as a means to tie the dependency on this OP (used in `fn.reinterpret`
        # below).
        xp = cp.get_array_module(input)
        res = xp.array(input.shape, dtype=xp.int32)
        return res

    # Run the check as a Python op and create a dependency without altering data or layout.
    checked_shape = fn.python_function(input, function=check_type_fn, num_outputs=1, preserve=True)
    # Tie the dependency by reinterpreting `input` with a shape argument derived from `checked`.
    # This keeps the original data and layout while ensuring the check op is executed. Note that compared
    # to directly passing in `input` through the python function, this approach:
    #  - Ensures that the layout of the input is preserved
    #  - Avoids copying large amounts of data (e.g. stacks of images)
    res = fn.reinterpret(input, shape=checked_shape.cpu())
    return res
