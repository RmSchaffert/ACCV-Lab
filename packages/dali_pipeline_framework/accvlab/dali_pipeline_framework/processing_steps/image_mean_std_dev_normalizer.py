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

from typing import Union, Sequence

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np

import nvidia.dali.fn as fn
import nvidia.dali.types as types

from ..pipeline.sample_data_group import SampleDataGroup

from .pipeline_step_base import PipelineStepBase


class ImageMeanStdDevNormalizer(PipelineStepBase):
    '''Normalize RGB or BGR images by mean and standard deviation, using pre-defined mean & standard deviation values.

    Normalization subtracts the mean and divides by the standard deviation per channel over spatial axes.
    Scalars broadcast to all channels; For 3‑vectors, each element corresponds to a channel; No distinction
    between RGB and BGR is made. This means that the mean and standard deviation values need to be provided
    for the channels in the order corresponding to the image format.

    Note:
        The mean and standard deviation values need to be provided on construction. They are not computed
        from the images at runtime.
    '''

    def __init__(
        self,
        image_name: Union[str, int],
        mean: Union[Sequence[float], float],
        std_dev: Union[Sequence[float], float],
        output_type: types.DALIDataType = types.DALIDataType.FLOAT,
    ):
        '''

        Args:
            image_name: Name of the image data fields to normalize.
            mean: Mean value used as basis for the normalization. Can be a single value (applied to all color channels) or a vector, containing the values for all channels.
            std_dev: Standard deviation used as basis for the normalization. Can be a single value (applied to all color channels) or a vector, containing the values for all channels.
            output_type: Data type for the output image. Default value is ``types.DALIDataType.FLOAT`` (i.e. 32-bit float).

        '''

        self._image_name = image_name
        np_type = SampleDataGroup.get_numpy_type_for_dali_type(output_type)

        if not isinstance(mean, Sequence) and not (isinstance(mean, np.ndarray) and len(mean.shape) == 0):
            mean = [mean] * 3
        if not isinstance(std_dev, Sequence) and not (
            isinstance(std_dev, np.ndarray) and len(std_dev.shape) == 0
        ):
            std_dev = [std_dev] * 3

        self._mean = np.expand_dims(np.array(mean, dtype=np_type), axis=[0, 1])
        self._std_dev = np.expand_dims(np.array(std_dev, dtype=np_type), axis=[0, 1])

        if not np.all(self._std_dev > 0):
            raise ValueError("Standard deviation must be greater than 0")

        self._output_type = output_type

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # Note that only the actual image normalization becomes part of the DALI graph, the search for images as well as resolvong the paths obtained in order to access the image,
        # and the change of the data type, happen at graph construction time and do not influence the run time.

        # Find all images
        image_paths = data.find_all_occurrences(self._image_name)

        for ip in image_paths:
            # Get the image
            image = data.get_item_in_path(ip)
            # Check if the data type changes
            datatype_changes = data.get_type_of_item_in_path(ip) != self._output_type
            # Convert image if needed
            if datatype_changes:
                image = fn.cast(image, dtype=self._output_type)
            # Normalize image
            image = fn.normalize(
                image,
                axes=[0, 1],
                mean=self._mean,
                stddev=self._std_dev,
                dtype=self._output_type,
                batch=False,
            )
            # Change the data type of the data field containing the image (if needed)
            if datatype_changes:
                data.change_type_of_data_and_remove_data(ip, self._output_type)
            # Set the normalized image
            data.set_item_in_path(ip, image)

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        image_paths = data_empty.find_all_occurrences(self._image_name)

        # Make sure at least 1 image is available
        if len(image_paths) == 0:
            raise KeyError(
                f"No occurrences of images found. Fields containing images are expected to have the name '{self._image_name}', as specified in the constructor."
            )

        # For each image, change the data type in the output to float32
        for ip in image_paths:
            data_empty.change_type_of_data_and_remove_data(ip, types.DALIDataType.FLOAT)

        return data_empty
