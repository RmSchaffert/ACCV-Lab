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

import time
import torch
from accvlab.optim_test_tools import NVTXRangeWrapper

# @NOTE
# In this example, the individual sections (such as Main Script, Code Part I, Code Part II, etc.)
# indicate parts of the code which, in the actual use case, would be in different files, and would rely
# on the `NVTXRangeWrapper` being a singleton to enable its use throughout the files as shown here.


# --------------------------- Main Script ---------------------------

nvtx_wrp = NVTXRangeWrapper()
# @NOTE
# To activate the NVTX wrapper, it needs to be enabled.
# If the following step is omitted, pushing and popping ranges will have no effect.
# Try commenting out the following call to `enable()`.
#
# Also, note that the `keep_track_of_range_order` parameter should be set to `False` during actual profiling,
# as it has an overhead and should only be enabled for debugging purposes.
nvtx_wrp.enable(
    sync_on_push=True,
    sync_on_pop=True,
    keep_track_of_range_order=True,  # Only set to `True` for debugging purposes (adds overhead)
)
# @NOTE
# Note that if the wrapper is not enabled, calling its methods has minimal overhead
# (call to an empty method).

# -------------------------------------------------------------------


# @NOTE
# If a code part (see below) is used in isolation (meaning that there is no other code which already
# enabled the NVTX wrapper), the wrapper will be disabled and any related calls will have no effect.
# The overhead for the wrapper is minimal in this case (call to an empty function).

num_iters = 16

# "Initialize" the GPU
torch.cuda.synchronize()

for i in range(num_iters):
    # --------------------------- Code Part I ---------------------------

    # @NOTE
    # This will not create a new instance, but re-use the instance created above.
    nvtx_wrp = NVTXRangeWrapper()
    nvtx_wrp.range_push("meas1")
    time.sleep(0.02)
    # ... continue and at some point call code part II

    # -------------------------------------------------------------------

    # --------------------------- Code Part II --------------------------

    nvtx_wrp = NVTXRangeWrapper()
    nvtx_wrp.range_push("meas2")
    time.sleep(0.05)
    nvtx_wrp.range_pop()
    # @NOTE
    # If the "unexpected range" range is pushed but not popped, then
    # `nvtx_wrp.range_pop("meas1")` (see below) will trigger an error.
    # Try uncommenting the "unexpected range" push below to see this.

    # >> nvtx_wrp.range_push("unexpected range")

    # ... continue and at some point call code part III

    # -------------------------------------------------------------------

    # -------------------------- Code Part III --------------------------

    # @NOTE
    # Here we want to check whether the range that we are popping is the expected one.
    # This can be done by specifying the expected range name when popping.
    # This will trigger an error if we pushed an "unexpected range" in Code Part II.
    #
    # Note that keeping track of the range stack internallyadds overhead and should only be enabled for
    # debugging purposes, not when actually performing the profiling (can be configured using the
    # `keep_track_of_range_order` parameter when calling `enable()` for the wrapper). If you set this
    # parameter to `False`, the mismatch check will be skipped (and no error will be raised even if the
    # "unexpected range" is pushed).
    nvtx_wrp.range_pop("meas1")
    # ... continue and at some point call code part IV

    # -------------------------------------------------------------------

    # -------------------------- Code Part IV --------------------------

    nvtx_wrp = NVTXRangeWrapper()
    if i % 3 == 2:
        nvtx_wrp.range_push("meas3")
        time.sleep(0.01)
        nvtx_wrp.range_pop()
    nvtx_wrp.range_push("meas2")
    time.sleep(0.01)
    nvtx_wrp.range_pop()

    # -------------------------------------------------------------------

print("Script ran without errors")
