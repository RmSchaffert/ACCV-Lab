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

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from nvidia.dali.data_node import DataNode

from nvidia.dali import types

from ..pipeline import SampleDataGroup


class CallableBase(ABC):
    '''Abstract base class for a callable class which can be used in the pipeline.

    Note that callables deriving from :class:`CallableBase` are expected to run with the DALI external source
    not in batch mode, i.e. return one sample at a time. This improves the distribution of the work onto the
    individual worker processes of the external source.

    Also see [1], and more specifically [2], for how an input callable class is used to load the input data
    into a DALI pipeline. The :meth:`__call__` operator is the interface that the DALI external source
    expects.

    Using a callable with the DALI parallel external source is more efficient than using an input iterable
    due to the possibility of distribution the work onto multiple workers instead of only running it async
    to the main thread, but still sequentially in a single worker.

    Note that an input callable must be stateless (see warning in [3]), which may make certain advanced
    sampling patterns more challenging to implement compared to an input iterable.

    Note:
        The :attr:`used_sample_data_structure` property is used by our pipeline to obtain the data format
        blueprint used for the input. Note that the actual output of a callable is the flattened data
        from this format (see :meth:`SampleDataGroup.get_data`), and the returned blueprint can be used
        to fill the data back into its structured form (see :meth:`SampleDataGroup.set_data`).

    Note:
        Note that ready-to use callable classes (:class:`ShuffledShardedInputCallable`,
        :class:`SamplerInputCallable`) are provided by this module and can be used in many cases,
        so that often there is no need to implement a custom callable.

    Note:
        Also see :class:`IterableBase` for an alternative to the callable interface. While the callable
        interface is potentially more efficient (allowing to distribute the work onto multiple workers),
        the iterable interface is more flexible as it is not expected to be stateless.

    Important:
        To be used with the DALI parallel external source, the callable needs to be serializable.
        If it contains any objects that cannot be serialized, these objects should not be created in the
        constructor, but rather created when the :meth:`__call__` method is called for the first time.
        At this point, the callable is already in the worker process, and therefore, it does not need
        to be serializable anymore.


    [1] https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/parallel_external_source.html

    [2] https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/parallel_external_source.html#Adjusting-to-Callable-Object

    [3] https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.external_source.html

    '''

    @property
    @abstractmethod
    def used_sample_data_structure(self) -> SampleDataGroup:
        '''Get the sample data format of the input.

        Get the blueprint (as defined in documentation of
        :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup`, i.e. a
        :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` object without any actual
        data but with the data format set up) describing the input data.

        Returns:
            :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` object describing the input data format.
        '''
        pass

    @abstractmethod
    def __call__(self, sample_info: types.SampleInfo) -> Tuple[DataNode, ...]:
        '''Get data of sample with the ID as described by sample_info.

        The returned data is expected to be a flattened sequence of the individual data fields contained
        in the :attr:`used_sample_data_structure`, i.e. if ``data_group`` is the :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup`
        object containing the data, then the output of this method should be ``data_group.get_data()``
        (see :meth:`SampleDataGroup.get_data() <accvlab.dali_pipeline_framework.pipeline.SampleDataGroup.get_data>` for more details).

        Args:
            sample_info: Info of the sample to provide the data for.

        Returns:
            The input data fields (as flat sequence).

        Raises:
            :exc:`StopIteration`: If the end of an epoch is encountered. Note that this is part of the
                normal behavior once the epoch is exhausted and is expected by the external source,
                and is not an error.

        '''
        pass

    @property
    @abstractmethod
    def length(self) -> Optional[int]:
        '''Length of the dataset (i.e. number of samples in one epoch).

        Providing the length is optional. If it is not implemented, this method still needs to be overridden.
        In this case, it has to indicate that the length is not available by returning ``None``.

        Returns:
            The number of samples or batches in the dataset, or ``None`` if not available.
        '''
        pass
