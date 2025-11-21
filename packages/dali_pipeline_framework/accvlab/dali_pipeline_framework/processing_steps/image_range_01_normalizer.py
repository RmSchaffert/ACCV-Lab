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

from typing import Union

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import nvidia.dali.fn as fn
import nvidia.dali.types as types

from .pipeline_step_base import PipelineStepBase
from ..pipeline.sample_data_group import SampleDataGroup


class ImageRange01Normalizer(PipelineStepBase):
    '''Convert RGB or BGR image from UINT8 to FLOAT and scale to [0.0, 1.0].

    Each matching image is cast to ``types.DALIDataType.FLOAT`` and divided by 255.0 per channel.
    '''

    def __init__(self, image_name: Union[str, int]):
        '''
        Args:
            image_name: Name of the image data field(s) to normalize.
        '''

        self._image_name = image_name

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # Note that only the actual image normalization becomes part of the DALI graph, the search for images as well as resolvong the paths obtained in order to access the image,
        # and the change of the data type, happen at graph construction time and do not influence the run time.

        # Find all images
        image_paths = data.find_all_occurrences(self._image_name)

        for ip in image_paths:
            # Get the image
            image = data.get_item_in_path(ip)
            # Normalize image
            image = fn.cast(image, dtype=types.DALIDataType.FLOAT) * (1.0 / 255.0)
            # Change the data type of the data field containing the image
            data.change_type_of_data_and_remove_data(ip, types.DALIDataType.FLOAT)
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
