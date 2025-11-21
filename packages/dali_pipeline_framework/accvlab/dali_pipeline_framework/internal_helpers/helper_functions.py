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

from typing import Sequence, Union, Any

import numpy as np
import nvidia.dali as dali
import nvidia.dali.types as types
import nvidia.dali.fn as fn


def get_as_data_node(value) -> dali.pipeline.DataNode:
    '''Convert a value to a DALI DataNode constant.

    Important:
        The result is always either an 32-bit integer or a 32-bit float
        (depending on the type of the input value).

    Args:
        value: The value to convert to a DALI DataNode.

    Returns:
        The value as a DALI DataNode.

    '''
    if not isinstance(value, dali.pipeline.DataNode):
        value_np = np.array(value)

        shape = value_np.shape

        is_bool = np.issubdtype(value_np.dtype, np.bool_)
        if is_bool:
            value_np = value_np.astype(np.int8)
            is_int = True
        else:
            is_int = np.issubdtype(value_np.dtype, np.integer)

        if is_int:
            if len(shape) > 0:
                idata = value_np.flatten().tolist()
                res = fn.constant(idata=idata, shape=shape, dtype=types.DALIDataType.INT32)
            else:
                res = fn.constant(idata=value_np.item(), shape=shape, dtype=types.DALIDataType.INT32)
        else:
            if len(shape) > 0:
                fdata = value_np.flatten().tolist()
                res = fn.constant(fdata=fdata, shape=shape, dtype=types.DALIDataType.FLOAT)
            else:
                res = fn.constant(fdata=value_np.item(), shape=shape, dtype=types.DALIDataType.FLOAT)
    else:
        res = value
    return res


def get_mapped(val: Union[Sequence, Any], mapping: dict, encapsulate: bool = False) -> list:
    '''Mapping of original values to numerical values.

    Handles nested lists/tuples recursively.

    The mapping is a dictionary with the original values (i.e. inputs to be mapped) as keys and the mapped
    values (i.e. the corresponding output values) as values. The dictionary may have a key ``None``, which
    is used as a default value if the input value is not in the mapping. If no ``None`` key is present,
    all the values must have a corresponding key in the mapping.

    Args:
        val: The value(s) to map.
        mapping: The mapping to use.
        encapsulate: Whether to encapsulate the result in a list.

    Returns:
        The mapped value(s).
    '''
    if isinstance(val, (list, tuple)):
        res = [get_mapped(v, mapping) for v in val]
    else:
        if val in mapping:
            res = mapping[val]
        else:
            res = mapping[None]
        if encapsulate:
            res = [res]
    return res
