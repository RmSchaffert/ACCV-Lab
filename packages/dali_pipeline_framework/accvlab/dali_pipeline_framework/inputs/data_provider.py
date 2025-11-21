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

from ..pipeline import SampleDataGroup


class DataProvider(ABC):
    '''Abstract base class for data providers.

    A data provider is an object that
      - Defines the data format of the samples
      - Provides samples from a dataset given sample indices

    It acts as an interface between the dataset and the DALI pipeline.

    To enable the use of a specific dataset with :class:`~accvlab.dali_pipeline_framework.pipeline.PipelineDefinition`
    as well as the included input callable & iterable classes, a corresponding data provider needs to be
    implemented.

    Important:
        Note that the data provider is not only specific to the dataset, but also specific to a
        use case (or a set of similar use cases), as it defines the data format of the individual samples.

        In simple cases, a single data provider can be parametrized for different use cases. However,
        in more complex cases, it is recommended to implement different data providers for different
        use cases, e.g. following the following approach:

          - Implement a data loader & data container class which are specific to the dataset
          - Implement a data conversion helper, which can be used by multiple data providers and performs
            repetitive tasks, e.g. converting the data to the correct format, obtaining the image data
            from individual files based on the loaded metadata, etc.
          - Implement use case-specific data providers, using the common functionality of the data loader,
            data container and conversion helper classes.

        This approach allows to keep the data provider class simple and focused on the specific use case,
        while being able to re-use the functionality which is specific to the used dataset but common to many
        use cases, e.g. the data loader, data container and conversion helper classes.
    '''

    @abstractmethod
    def get_data(self, sample_index: int) -> SampleDataGroup:
        '''Get the data for a given sample index.

        Args:
            sample_index: The index of the sample to get the data for.

        Returns:
            The data for the given sample index.
        '''
        pass

    @abstractmethod
    def get_number_of_samples(self) -> int:
        '''Get the number of samples in the dataset.

        Note:
            The number of samples in the dataset not necessarily the number of samples in one epoch, as
            e.g. some samples might be skipped or repeated to ensure full batches. Here,
            the actual number of samples in the dataset is returned.

        Note:
            The number of samples depends on the use case (e.g. if the dataset contains images from multiple
            camera views, is the number of samples the total number of images, or do multiple views need to be
            loaded for each sample? etc.).

        Returns:
            The number of samples in the dataset.
        '''
        pass

    @property
    @abstractmethod
    def sample_data_structure(self) -> SampleDataGroup:
        '''Get the data structure of the samples.

        The data structure is a blueprint :class:`SampleDataGroup` that defines the structure
        of the data without containing the actual data.

        Returns:
            The data structure of the samples.
        '''
        pass
