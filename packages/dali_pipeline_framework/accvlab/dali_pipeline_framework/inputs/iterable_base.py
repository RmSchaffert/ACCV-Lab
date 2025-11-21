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
from typing import Optional

from ..pipeline import SampleDataGroup


class IterableBase(ABC):
    '''Abstract base class for an iterable class which can be used in the pipeline.

    Classes derived from :class:`IterableBase` are expected to run in the DALI external source in batch-mode,
    i.e. returning one batch at a time.

    Also see [1] for how an input iterable class is used to load the input data into a DALI pipeline.

    Iterables are more flexible than callables as they can have an internal state, which is not possible for
    callables. However, they are less efficient than callables as they only allow to distribute the work onto
    a single worker when using the DALI parallel external source.

    Note:
        The :attr:`used_sample_data_structure` property is used by our pipeline to obtain the data format
        blueprint used for the input. Note that the actual output of a callable is the flattened data
        from this format (see
        :meth:`SampleDataGroup.get_data() <accvlab.dali_pipeline_framework.pipeline.SampleDataGroup.get_data>`), and the
        returned blueprint can be used
        to fill the data back into its structured form (see
        :meth:`SampleDataGroup.set_data() <accvlab.dali_pipeline_framework.pipeline.SampleDataGroup.set_data>`).

    Note:
        A ready-to-use :class:`SamplerInputIterable` is provided. Also see :class:`SamplerInputCallable`
        and :class:`ShuffledShardedInputCallable` for more ready-to-use options.

    Note:
        Also see :class:`CallableBase` for an alternative to the iterable interface. Note that the callable
        interface is potentially more efficient than the iterable interface (as it allows to distribute the
        work onto multiple workers), and should be preferred in general. However, for use cases requiring the
        input object to have an internal state, the iterable interface needs to be used as callables are
        expected to be stateless.

    Important:
        To be used with the DALI parallel external source, the iterable needs to be serializable.
        If it contains any objects that cannot be serialized, these objects should not be created in the
        constructor, but rather created when the :meth:`__next__` method is called for the first time.
        At this point, the iterable is already in the worker process, and therefore, it does not need
        to be serializable anymore.

    [1] https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html#Define-the-Data-Source

    '''

    @property
    @abstractmethod
    def used_sample_data_structure(self) -> SampleDataGroup:
        '''Sample data format of the input.

        Get the blueprint (as defined in documentation of
        :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup`, i.e. a
        :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` object without any actual data but with the
        data format set up) describing the input data.

        Returns:
            :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` object describing the input data format
        '''
        pass

    @abstractmethod
    def __iter__(self) -> 'IterableBase':
        '''Get the iterator (can be the same object as self) starting from the beginning.

        Returns:
            The iterator starting from the beginning.
        '''
        return self

    @abstractmethod
    def __next__(self) -> tuple:
        '''Get the next batch of data.

        The data is a flattened sequence of data set according to the data format described by
        :attr:`used_sample_data_structure` and then flattened. This means that
        ``self.used_sample_data_structure.set_data(self.__next__())`` would return a
        :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` with the correct input data format and filled
        with the actual data.

        Note:
            A flat sequence is returned here as this is the format expected by the DALI external source,
            which will use this object. The flat sequence can be obtained by calling
            :meth:`SampleDataGroup.get_data() <accvlab.dali_pipeline_framework.pipeline.SampleDataGroup.get_data>` on
            the :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` object containing the
            input data (according to :attr:`used_sample_data_structure`).

        Returns:
            The input data fields (as flat sequence)

        Raises:
            StopIteration: When there are no more batches to provide. Note that this is part of the
                normal behavior once the epoch is exhausted and is expected by the external source,
                and is not an error.

        '''
        pass

    @property
    @abstractmethod
    def length(self) -> Optional[int]:
        '''Length of one epoch.

        Providing the length is optional. If it is not implemented, this method still needs to be overridden.
        In this case, it has to indicate that the length is not available (by returning ``None``).
        Returns:
            The number of batches in the epoch, or ``None`` if not available.
        '''
        pass
