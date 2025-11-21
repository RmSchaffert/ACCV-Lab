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

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

import numpy as np

max_batch_size = 4


@pipeline_def
def np_random_numbers_pipeline():
    '''
    Non-DataNode expressions are evaluated at graph construction time (if not converted, which may e.g. happen automaticalle, as will be discussed below for if statements).
    This means that in this example, "np.random.uniform(...)" is evaluated at graph construction time and "data" becomes a constant during graph execution time, and each sample
    receives the same random numbers (check the output of the pipeline).
    '''
    # Non data-node expressions are evaluated at graph construction time (with some exceptions).
    # This means that "data becomes a constant durting graph execution time"
    data = np.random.uniform(low=0.0, high=1.0, size=(5,))
    return data


@pipeline_def
def dali_random_numbers_pipeline():
    '''
    This is the/a correct way to generate random numbers in each sample individually. The call to fn.random.uniform(...)
    becomes part of the graph and is run for each sample, generating new random numbers
    '''
    data = fn.random.uniform(range=[0.0, 1.0], shape=(5,))
    return data


@pipeline_def
def python_func_operator_random_numbers_pipeline():
    '''
    Python code can also be executed during graph run-time. However, this needs to be done inside a python function operator.
    This also means that the inputs/outputs to the python function are DALI-datatypes (tensors and tensor lists), which are available
    as numpy or cupy types inside the function. Here is an example of random number generation, which is executed at runtime.
    Note that while here, lists are directly used as inputs to the python function operator, they are internally converted to DALI types when calling
    the python_function operator and then to numpy/cupy arrays (depending on thether the CPU or GPU is used) when actually running the function code.
    Similarly, the returned results are converted to DALI types (show as DataNode at construction time)
    '''

    def function_to_use(range, shape):
        # Note that the inputs are converted to DALI types (tensors) when passed into the python_function operator and then to numpy/cupy
        # arrays when accessed from inside the function. Asserts demonstrate that here, indeed numpy arrays are present although we used list when calling the
        # operator
        assert isinstance(range, np.ndarray)
        assert isinstance(shape, np.ndarray)
        res = np.random.uniform(low=range[0], high=range[1], size=shape).astype(np.float32)
        return res

    data = fn.python_function([0.0, 1.0], 5, function=function_to_use)
    print(
        "########## Note that the result is a DataNode (at construction time). Output of str(data):\n"
        + str(data)
        + "\n#########"
    )
    return data


@pipeline_def
def if_pipeline_constr_time():
    '''
    If non-DataNode conditions are used in if statements, the branching happens at graph construction time and will be not part of the graph
    '''
    # Note that the condition is not a DataNode here. This means the branching happens at construction time
    condition = np.random.uniform(low=0.0, high=1.0) < 0.5
    if condition:
        test_var = 1.0
    else:
        test_var = 2.0

    return test_var


@pipeline_def
def if_pipeline():
    '''
    Conditionals are allowed in the graph and any variables assigned in branches become data nodes.
    '''
    # Note that the condition is a DataNode here. If it is not, the branching happens at construction time
    condition = fn.random.uniform(range=[0.0, 1.0]) < 0.5
    if condition:
        test_var = 1.0
    else:
        test_var = 2.0
    # 'test_var' is automatically ocnverted to DataNode as it is assigned inside the branches
    print("####################### Note the data type of str(test_var), which is DataNode: " + str(test_var))

    return test_var


@pipeline_def
def if_pipeline_alt():
    '''
    Conditionals are allowed in the graph and any variables assigned in branches become data nodes.
    Variables which are assigned in the branches need to be present in each branch. However, this is also
    fulfilled if they are defined outside the loop and only modified in some branches
    '''
    condition = fn.random.uniform(range=[0.0, 1.0]) < 0.5
    test_var = 2.0
    if condition:
        test_var = 1.0
    else:
        pass
    print("####################### Note the data type of str(test_var), which is DataNode: " + str(test_var))

    return test_var


@pipeline_def
def loop_error1():
    '''
    Loops cannot be used in the graph. In this case, the rror is that "num_iter" cannot be used as an integer to define the range
    '''
    num_iter = fn.random.uniform(range=[10, 20], dtype=types.DALIDataType.INT32)

    res = 0
    for i in range(num_iter):
        res += i

    return res


@pipeline_def
def loop_error2():
    '''
    Loops cannot be used in the graph. In this case, the error is that DataNode is not iterable
    '''
    vals = fn.random.uniform(range=[10, 20], shape=10, dtype=types.DALIDataType.INT32)

    res = 0
    for v in vals:
        res += v

    return res


@pipeline_def
def loop_at_construction_time():
    '''
    However, loops are ok at graph construction time. Here, iterations are defined using non-DataNode types (as these are
    evaluated at graph construction time). Here, we mock a case where the maximum number of iterations can be given
    at graph construction time, but the actual needed number of iterations is smaller. This can be resolved using an if-statement
    (using DataNodes as condition)
    '''
    max_num_iter = 20
    vals = fn.random.uniform(range=[10, 20], shape=(10,), dtype=types.DALIDataType.INT32)

    # HEre, we know that 'num_real_iter == 10'. However, in a real use case, 'num_real_iter' may depend e.g. on the date stored for each sample such as number of objects visible there
    # and cannot be set at ocnstruction time.
    num_real_iter = fn.shapes(vals)[0]

    res = 0
    for i in range(max_num_iter):
        if i < num_real_iter:
            res += vals[i]
    return vals, res


@pipeline_def
def construction_time_vs_run_time_operations_subtle_error():
    '''
    Note that the above example (loop_at_construction_time) works correctly as the maximum available index of "vals" is checked at run time ('i < num_real_iter') and 'val[i]' is also accessed at run time.
    In this example, there is an additional variable 'consts', but it is a numpy array and therefore 'consts[i]' is evaluated at construction time (and becomes a constant in the graph).
    At construction time, both branches are evaluated (i.e. graph is constructed for both branches). This means that 'consts[i]' is accessed up to
    'i == max_num_iter - 1 (==19 )' during graph construction and an out of bounds error occurs as the size is 10. To fix this, another if statement needs to be added which checks i against the
    size of consts.
    Combining these if statements into one is possible, but tricky. Whether branching happends at construction time or run time depends on what type the condition evaluates to,
    and this may be either a simple boolean (construction time) or DataNode, depending on the involved variables and making use of the lazyness of the logical operators.
    E.g. 'i < consts.shape[0] and i < num_real_iter' would evaluate to False for 'i > 9' at construction time (-> no OOB error), while 'i < num_real_iter and i < consts.shape[0]' would evaluate to a DataNode
    at construction time, and both branches would be processed at construction time (--> OOB error).
    '''

    max_num_iter = 20

    vals = fn.random.uniform(range=[10, 20], shape=(10,), dtype=types.DALIDataType.INT32)
    consts = np.random.uniform(low=10, high=20, size=(10,)).astype(np.int32)

    num_real_iter = fn.shapes(vals)[0]

    res = 0
    for i in range(max_num_iter):
        # Subtle error: both branches are processed on construction time to construct the graph, and this will lead to trying accessing consts[i] for i >= num_real_iter, even if
        # only the false branch is ever executed at run time
        if i < num_real_iter:
            res += vals[i] + consts[i]
    return vals, res


@pipeline_def
def loop_at_construction_time_break_error():
    '''
    While loops are ok at graph construction time, the number of iterations needs to be known at graph construction time. This means
    that the iterations cannot be controlled using DataNode variables. Here, we try to do this by executing 'break' based on a DataNode value.
    We get the error 'TypeError: "DataNode" was used in conditional context'. While if statements may have conditions of the type DataNode,
    this is not possible e.g. with 'break', as 'break' would need to be executed at graph construction time, while DataNode represents a value which is
    computed during graph execution time and the branching happens at graph execution time.
    '''

    max_num_iter = 20
    vals = fn.random.uniform(range=[10, 20], shape=(10,), dtype=types.DALIDataType.INT32)

    # Although here, 'num_real_iter' is theoretically known at construction time, it could be dynamic if the shape of vals e.g. depends on e.g. the input image size
    num_real_iter = fn.shapes(vals)[0]

    res = 0
    for i in range(max_num_iter):
        if i >= num_real_iter:
            break
        res += vals[i]
    return vals, res


@pipeline_def
def loop_at_construction_time_inconsistent_tensor_location_error():
    '''
    One needs to keep the location of the tensors consistent. If depending on a condition (which is evaluated at graph run-time), the location of the tensor is either on the CPU or the GPU,
    this will lead to errors. In this example, it cannot be determined at construction time if for the individual iterations, 'data' is located on the CPU or the GPU.
    '''
    max_num_iter = 20

    data = fn.constant(fdata=0.0, shape=(1,), dtype=types.DALIDataType.FLOAT, device="cpu")

    for i in range(max_num_iter):
        if fn.random.uniform(range=[0.0, 1.0], dtype=types.DALIDataType.FLOAT) < 0.5:
            data = data.gpu()
            data += 1.0
        else:
            data += 2.0

    return data


@pipeline_def
def loop_at_construction_time_changing_tensor_location():
    '''
    Changing the tensor location in the loop is not a problem if the location in each iteration can be deduced correctly at graph construction time, i.e. changed based on non-DataNode conditions.
    '''
    max_num_iter = 20

    data = fn.constant(fdata=0.0, shape=(1,), dtype=types.DALIDataType.FLOAT, device="cpu")

    for i in range(max_num_iter):
        if i > 5:
            data = data.gpu()
            data += 1.0
        else:
            data += 2.0
    return data


@pipeline_def
def unused_subgraph_pruning():
    '''
    Unused parts of the graph are pruned. This means that not all code (that is on the executed branches) is neccessarily executed. In this example,
    'res_oob_unused = data[100]' would lead to an error. However, it is pruned away as its value is not used in the final output. Try returning
    'res_oob_unused_processed' or 'res_combined': In this case, the sub-graph containing the error is not pruned and the error will be raised.
    One needs to be careful that if a change leads to an error, the error may be at the changed location in the code or it may be in parts of the graph
    which were not used before but are now used (e.g. suppose 'res_combined' is returned and its definition is changed from 'res_combined = 2.0 * res_ok_processed' to
    'res_combined = 2.0 * res_ok_processed + 3.0 * res_oob_unused_processed'). This will trigger the error, although the problematic code is not on the changed line.
    '''
    data = fn.constant(fdata=0.0, shape=(5,), dtype=types.DALIDataType.FLOAT, device="cpu")

    res_ok = data[0]

    # This gets pruned away as it is finally not needed to compute the final result
    res_oob_unused = data[100]

    res_ok_processed = res_ok + 20.0

    # This gets pruned away as it is finally not needed to compute the final result
    res_oob_unused_processed = res_oob_unused + 30.0

    # This gets pruned away as it is finally not needed to compute the final result
    res_combined = 2.0 * res_ok_processed + 3.0 * res_oob_unused_processed

    return res_ok_processed


@pipeline_def
def if_pipeline_use_dict():
    '''
    Variables which are changed in branches of the graph are automatically converted to DataNode types (tensors / tensor lists at graph run time).
    However, one can e.g. use dictionaries as long as the keys do not change. The actual values stored in the dictionaries are converted to DataNode types
    instead in this case.
    '''

    dict_var = {"key1": 1.0, "key2": 2.0}

    if fn.random.uniform(range=[0.0, 1.0], dtype=types.DALIDataType.FLOAT) < 0.5:
        dict_var = {"key1": 4.0, "key2": 5.0}

    print("Content of the dict element 'key1': " + str(dict_var["key1"]))

    return dict_var["key1"]


@pipeline_def
def if_pipeline_use_dict_error():
    '''
    Variables which are changed in branches of the graph are aotomatically converted to DataNode types (tensors / tensor lists at graph run time).
    However, one can e.g. use dictionaries as long as the keys do not change. In this example, the keys do change and this leads to an error.
    '''

    dict_var = {"key1": 1.0, "key2": 2.0}

    if fn.random.uniform(range=[0.0, 1.0], dtype=types.DALIDataType.FLOAT) < 0.5:
        dict_var = {"key1": 4.0, "key3": 5.0}

    print("Content of the dict element 'key1': " + str(dict_var["key1"]))

    return dict_var["key1"]


if __name__ == "__main__":

    # Replace if_pipeline_use_dict by any other pipeline definition from above to run the respective pipeline
    pipe = dali_random_numbers_pipeline(
        batch_size=max_batch_size, num_threads=1, device_id=0, enable_conditionals=True
    )
    pipe.build()

    num_batches = 1

    for i in range(num_batches):

        print(f" --------- {i}th batch ----------")
        pipe_out = pipe.run()
        print(" - output of the pipeline -")

        print(pipe_out)
