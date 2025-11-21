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

import threading
from typing import Optional

import torch

if __name__ != "__main__":
    from .singleton_base import SingletonBase
else:
    from singleton_base import SingletonBase


class NVTXRangeWrapper(SingletonBase):
    """Wrapper for NVTX ranges.

    This is a singleton class which allows for enabling the use of NVTX ranges and configuring how the ranges
    are used from any part of the implementation.

    The wrapper must be first enabled before any measurements are performed. If not enabled, calls to any
    methods have minimal overhead. Enabling can be done from any part of the code (as this is a singleton).

    Compared to using the NVTX range push/pop functionality directly, it offers the following advantages:

    - It is possible to easily configure whether CUDA synchronization is performed when pushing/popping a
      range. The synchronization is part of the push/pop methods and so can be turned on and off without
      changes to the code where the ranges are used, and is not performed if not needed.
    - If not enabled, calls to push/pop have minimal overhead (call to an empty function). Note that while the
      pushing/popping of ranges itself also has negligible overhead using NVTX directly, profiling-related
      CUDA synchronizations need to be handled manually in this case.
    - Range mismatch checks: The wrapper allows for checks whether the popped range corresponds to the range
      that is expected to be popped. This functionality can be turned on or off as part of the configuration
      when enabling the wrapper. This functionality has an overhead, and so should be only enabled for
      debugging purposes, and be turned off when actual profiling is performed.
    """

    def __init__(self, *args, **kwargs):
        """

        Note:
            When obtaining an object using (``NVTXRangeWrapper()``) the singleton is returned if already
            created.

            If parameters are provided when calling ``NVTXRangeWrapper()``, this will enable the NVTX range
            wrapper (equivalent to calling :meth:`enable`).
            Note that enabling can only be done once, and will lead to an error if attempted a second time.

        Args:
            sync_on_push: Whether to synchronize the CUDA device every time before pushing a range
            sync_on_pop: Whether to synchronize the CUDA device every time before popping a range
            keep_track_of_range_order: Whether to keep track of the range stack internally. A range name may
                be specified optionally when popping a range, and a check is performed whether the popped range
                corresponds to the range that is expected to be popped if this is set to ``True``. Note
                that this has an overhead and so should be only enabled for debugging purposes, and be turned
                off when performing the actual profiling.
        """

        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._enabled = False
            self._SET_DOCSTRINGS_OF_ENABLED_METHOD_VARIANTS()
        if len(args) > 0 or len(kwargs) > 0:
            self.enable(*args, **kwargs)

    def enable(self, sync_on_push: bool, sync_on_pop: bool, keep_track_of_range_order: bool):
        """Enable the NVTX range wrapper.

        This method can be called only once and enables the NVTXRangeWrapper singleton.
        Any use of the singleton before enabling it is ignored.

        Args:
            sync_on_push: Whether to synchronize the CUDA device every time before pushing a range
            sync_on_pop: Whether to synchronize the CUDA device every time before popping a range
            keep_track_of_range_order: Whether to keep track of the range stack internally. A range name may
                be specified optionally when popping a range, and a check is performed whether the popped range
                corresponds to the range that is expected to be popped if this is set to ``True``. Note
                that this has an overhead and so should be only enabled for debugging purposes, and be turned
                off when performing the actual profiling.
        """
        assert not self._enabled, (
            "NVTXRangeWrapper singleton can be enabled only once (either by passing arguments when it "
            "is obtained, or by calling :meth:`enable` directly)"
        )
        self._enabled = True
        if sync_on_push:
            self._maybe_sync_push = lambda: torch.cuda.synchronize()
        else:
            self._maybe_sync_push = lambda: None
        if sync_on_pop:
            self._maybe_sync_pop = lambda: torch.cuda.synchronize()
        else:
            self._maybe_sync_pop = lambda: None
        self._keep_track_of_range_order = keep_track_of_range_order
        if keep_track_of_range_order:
            self._range_stacks = {}

        # Set the methods
        self.range_push = self._range_push_enabled
        self.range_pop = self._range_pop_enabled

    @property
    def is_enabled(self) -> bool:
        '''Whether the NVTXRangeWrapper is enabled'''
        return self._enabled

    def range_push(self, range_name: str):
        """Push a NVTX range

        Args:
            range_name: Range name
        """
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    def range_pop(self, range_name: Optional[str] = None):
        """Pop a NVTX range and optionally check if the popped range
        is the expected range to be popped.

        Note that the check is performed only if configured to be used
        when calling `enable()`.

        Args:
            range_name: Range name. If set, will be used to check whether the popped
                range name corresponds to the given name and raise an assertion
                error if not.
        """
        # Empty method to minimize overhead if not enabled. Will be replaced when enabling.
        pass

    @classmethod
    def _SET_DOCSTRINGS_OF_ENABLED_METHOD_VARIANTS(cls):
        """Set the docstrings of the enabled method variants.

        This is done to ensure that the correct docstring is present in the methods
        once the NVTXRangeWrapper is enabled, and the original (disabled) methods are
        replaced by the
        enabled variants.
        """
        cls._range_push_enabled.__doc__ = cls.range_push.__doc__
        cls._range_pop_enabled.__doc__ = cls.range_pop.__doc__

    def _range_push_enabled(self, range_name):
        """TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested
        for the first time.
        """
        self._maybe_sync_push()
        torch.cuda.nvtx.range_push(range_name)
        if self._keep_track_of_range_order:
            thread_id = threading.get_native_id()
            if not thread_id in self._range_stacks:
                self._range_stacks[thread_id] = []
            self._range_stacks[thread_id].append(range_name)

    def _range_pop_enabled(self, range_name=None):
        """TEMPORARY DOCSTRING
        This is the enabled variant of the corresponding method (same name without leading `_` and training `_enabled`).
        This docstring will be replaced with the docstring of the corresponding method when an instance is requested for
        the first time.
        """
        self._maybe_sync_pop()
        torch.cuda.nvtx.range_pop()
        if self._keep_track_of_range_order:
            thread_id = threading.get_native_id()
            assert (
                thread_id in self._range_stacks and len(self._range_stacks[thread_id]) > 0
            ), "No ranges are present for the thread in which a range is being popped"
            last_range = self._range_stacks[thread_id].pop()
            if range_name is not None:
                assert range_name == last_range, (
                    f"Popped range (`{last_range}`) does not correspond to expected range to pop (`{range_name}`)\n"
                    f"  Range stack for thread {thread_id} before the pop is: {self._range_stacks[thread_id] + [last_range]}"
                )


if __name__ == "__main__":

    import time

    # --------------------------- Main Script ---------------------------
    nvtx_wrp = NVTXRangeWrapper()
    # To activate the nvtx wrapper, it needs to be enabled.
    # If the following step is omitted, pushing and popping ranges will have no effect.
    # Try commenting out the following line:
    nvtx_wrp.enable(True, True, True)
    # Note that if the wrapper is not enabled, calling its methods has minimal overhead
    # (call to an empty method).
    # -------------------------------------------------------------------

    # Note that if a code part (see below) is used in isolation, the actual functionality will be disabled and
    # no error will occur. The overhead for the wrapper is minimal in this case (call to an empty function).

    num_iters = 16

    # "Initialize" the GPU
    torch.cuda.synchronize()

    start = time.time()
    for i in range(num_iters):
        # --------------------------- Code Part I ---------------------------
        # This will not create a new instance, but re-use the instance created above.
        # No configuration should be done (passing parameters will raise error).
        nvtx_wrp = NVTXRangeWrapper()
        nvtx_wrp.range_push("meas1")
        time.sleep(0.02)
        # ... continue and at some point call code part II ...
        # -------------------------------------------------------------------

        # --------------------------- Code Part II --------------------------
        nvtx_wrp = NVTXRangeWrapper()
        nvtx_wrp.range_push("meas2")
        time.sleep(0.05)
        nvtx_wrp.range_pop()
        # NOTE: If the following range is pushed but not popped, then
        # `nvtx_wrp.range_pop("meas1")` (see below) will trigger an error
        nvtx_wrp.range_push("unexpected range")
        # -------------------------------------------------------------------

        # ------------------------ Code Part I (back) -----------------------
        # ... back from the call to code part II ...
        # Here we want to check whether the range that we are popping is the expected one
        nvtx_wrp.range_pop("meas1")
        # -------------------------------------------------------------------

        # -------------------------- Code Part III --------------------------
        nvtx_wrp = NVTXRangeWrapper()
        # Measurements can be performed in only some of the iterations
        if i % 3 == 2:
            nvtx_wrp.range_push("meas3")
            time.sleep(0.01)
            nvtx_wrp.range_pop()
        # We can perform a measurement for the same entry multiple times in a
        # single iteration. This can be in the same or different code parts.
        nvtx_wrp.range_push("meas2")
        time.sleep(0.01)
        nvtx_wrp.range_pop()
        # -------------------------------------------------------------------
    end = time.time()
    print(f"Time: {end - start}")
