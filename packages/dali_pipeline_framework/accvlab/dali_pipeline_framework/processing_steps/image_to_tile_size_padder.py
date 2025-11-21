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
from collections.abc import Sequence as ABCSequence

import nvidia.dali.fn as fn
import nvidia.dali.types as types

from .pipeline_step_base import PipelineStepBase
from ..pipeline.sample_data_group import SampleDataGroup


class ImageToTileSizePadder(PipelineStepBase):
    '''Pad images so height and width are multiples of a given tile size.

    The image is padded with zeros and the image size field is updated to the padded size.
    '''

    def __init__(
        self,
        image_name: Union[str, int],
        tile_size_to_pad_to: Union[int, Sequence[int]],
    ):
        '''

        Args:
            image_name: Name of the image data fields to pad.
            tile_size_to_pad_to: Tile size to be used. This means that the size of the padded image will be a multiple of the tile size (in each dimension).
        '''

        self._image_name = image_name
        self._tile_size_to_pad_to = (
            tile_size_to_pad_to
            if isinstance(tile_size_to_pad_to, ABCSequence)
            else [tile_size_to_pad_to, tile_size_to_pad_to]
        )
        assert (
            self._tile_size_to_pad_to[0] > 0 and self._tile_size_to_pad_to[1] > 0
        ), "Tile size must be greater than 0. To retain the original image size, use tile size 1."

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # Note that only the actual image normalization becomes part of the DALI graph, the search for images as well as resolvong the paths obtained in order to access the image,
        # and the change of the data type, happen at graph construction time and do not influence the run time.

        # Find all images
        image_paths = data.find_all_occurrences(self._image_name)

        for ip in image_paths:
            # Get the image
            parent = data.get_parent_of_path(ip)
            image = parent[self._image_name]

            # Extract the image size dynamically using .shape() method
            image_shape = image.shape()
            # Use fn.stack to create a proper tensor with [height, width]
            # Cast to int32 for compatibility with padding operations
            image_hw = fn.cast(fn.stack(image_shape[-3], image_shape[-2]), dtype=types.DALIDataType.INT32)

            # Calculate the padded size
            res_hw = self._get_padded_size(image_hw)

            # Pad the image
            res_image = fn.pad(image, shape=fn.stack(res_hw[0], res_hw[1]), axes=[-3, -2])

            # Set the padded image
            parent[self._image_name] = res_image
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        '''Check the input data format for compatibility with this step and provide the output data format (blueprint) given the input data format for the specific step.

        Please also see the documentation of the base class :class:`PipelineStepBase` for more details.

        Args:
            data_empty: Input data format (blueprint)

        Returns:
            Resulting data format (blueprint)
        '''
        image_paths = data_empty.find_all_occurrences(self._image_name)

        # Make sure at least 1 image is available
        if len(image_paths) == 0:
            raise KeyError(
                f"No occurrences of images found. Fields containing images are expected to have the name '{self._image_name}', as specified in the constructor."
            )

        return data_empty

    def _get_padded_size(self, image_hw: Sequence[int]) -> Sequence[int]:
        '''Get the padded size of an image.

        Args:
            image_hw: Image size in format: [height, width].

        Returns:
            Padded size in format: [height, width].
        '''
        size_y = (
            (image_hw[0] + self._tile_size_to_pad_to[1] - 1) // self._tile_size_to_pad_to[1]
        ) * self._tile_size_to_pad_to[1]

        size_x = (
            (image_hw[1] + self._tile_size_to_pad_to[0] - 1) // self._tile_size_to_pad_to[0]
        ) * self._tile_size_to_pad_to[0]

        res_hw = fn.stack(size_y, size_x)
        return res_hw
