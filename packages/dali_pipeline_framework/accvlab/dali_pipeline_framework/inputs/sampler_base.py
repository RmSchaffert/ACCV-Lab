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
from typing import List, Optional


class SamplerBase(ABC):
    '''Abstract base class for samplers that provide indices for data loading.

    A sampler is responsible for determining which samples from a dataset should be
    included in each batch during training. It can be epoch-based (where epochs have
    clear boundaries) or continuous (where sampling continues indefinitely).

    A sampler can be used with either :class:`SamplerInputIterable` or :class:`SamplerInputCallable`.
    Please also see the documentation of these classes.

    Note:
        Samplers can be used for complex sampling strategies, e.g. for sampling of sequences. For this,
        a :class:`SequenceSampler` class is provided, which can be used to sample consecutive samples
        (for each sample index ``i`` in consecutive batches) from a set of sequences. See the documentation of
        the sequence sampler for more details.

        For simple use-cases, a sampler may not be required.
        A :class:`ShuffledShardedInputCallable` class is provided, which can be used for random
        sampling without the need for a sampler implementation.

        Before implementing a custom sampler, consider whether the available ready-to-use solutions can be
        used.

    Important:
        To be used with :class:`SamplerInputIterable`, the sampler needs to be serializable (see the
        corresponding note in the documentation of :class:`IterableBase`). If the sampler contains any
        objects that cannot be serialized (e.g. generators), these objects should not be created in the
        constructor, but rather created when the :meth:`get_next_batch_indices` method is called for the
        first time. At this point, the iterable is already in the worker process, and therefore, the sampler
        does not need to be serializable anymore.

        Note that the :class:`SamplerInputCallable` does not require the sampler to be serializable as
        it is only used to generate the look-up table in advance. However, it is advisable to keep sampler
        objects compatible with both :class:`SamplerInputIterable` and :class:`SamplerInputCallable`, and
        therefore, to not create non-serializable objects before the first call to
        :meth:`get_next_batch_indices`.
    '''

    @abstractmethod
    def get_next_batch_indices(self) -> List[int]:
        '''Get the indices for the samples in the next batch.

        If the sampler is epoch-based and the next batch is not inside the current epoch,
        :exc:`StopIteration` shall be raised instead of returning data.
        In this case, a call to :meth:`reset` indicates the start of the next epoch. After :meth:`reset`
        is called, :meth:`get_next_batch_indices` shall continue with returning the indices for
        the newly started epoch.

        Returns:
            List of sample indices for the next batch.

        Raises:
            StopIteration: If the sampler is epoch-based and the current epoch has ended. Note that this
                is part of the normal behavior once the epoch is exhausted and is expected by the external
                source, and is not an error.
        '''
        pass

    @property
    @abstractmethod
    def is_epoch_based(self) -> bool:
        '''Indicate whether the sampling is epoch-based.

        Returns:
            ``True`` if the sampler is epoch-based, ``False`` otherwise.
        '''
        pass

    @abstractmethod
    def reset(self):
        '''Start a new epoch.

        This method should be called to reset the sampler state and begin a new epoch.
        Only applicable for epoch-based samplers.
        '''
        pass

    @property
    @abstractmethod
    def length(self) -> Optional[int]:
        '''Length of one epoch.

        Providing the length is optional. If it is not implemented, this method still needs to be
        overridden. In this case, it has to indicate that the length is not available (by returning
        ``None``).

        Returns:
            The number of samples in the epoch, or ``None`` if not available.
        '''
        pass
