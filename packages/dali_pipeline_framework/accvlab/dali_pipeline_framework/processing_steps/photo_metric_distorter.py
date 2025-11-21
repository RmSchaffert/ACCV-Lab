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

import nvidia.dali.fn as fn
import nvidia.dali.math as dali_math
import nvidia.dali.types as types
from nvidia.dali.pipeline import DataNode

from ..pipeline.sample_data_group import SampleDataGroup

from .pipeline_step_base import PipelineStepBase


class PhotoMetricDistorter(PipelineStepBase):
    '''Apply photometric augmentations to images (brightness, contrast, saturation, hue, channel swap).

    The same random decision & parametrization for each augmentation is shared across all matched images to
    keep consistency (e.g., across multi-view inputs).
    '''

    def __init__(
        self,
        image_name: Union[str, int],
        min_max_brightness: Sequence[float],
        min_max_hue: Sequence[float],
        min_max_contrast: Sequence[float],
        min_max_saturation: Sequence[float],
        prob_brightness_aug: float = 0.5,
        prob_hue_aug: float = 0.5,
        prob_contrast_aug: float = 0.5,
        prob_saturation_aug: float = 0.5,
        prob_swap_channels: float = 0.5,
        is_bgr: bool = False,
        enforce_process_on_gpu: bool = True,
    ):
        '''

        Args:
            image_name: Name of the image data fields to augment.
            min_max_brightness: Minimum and maximum biases to apply to the brightness. Note that as the
                image may be in different ranges ([0; 1] for float images, [0; 255] for uint8 images), the
                values provided here are expected to be in the corresponding range.
            min_max_hue: Minimum and maximum change in hue (degrees).
            min_max_contrast: Minimum and maximum contrast factor (multiplicative).
            min_max_saturation: Minimum and maximum saturation factor (multiplicative in HSV space).
            prob_brightness_aug: Probability to apply brightness augmentation. Default value is 0.5.
            prob_hue_aug: Probability to apply hue change augmentation. Default value is 0.5.
            prob_contrast_aug: Probability to apply contrast augmentation. Default value is 0.5.
            prob_saturation_aug: Probability to apply saturation augmentation. Default value is 0.5.
            prob_swap_channels: Probability to randomly permute color channels.
            is_bgr: Whether the image is in BGR format (RGB otherwise).
            enforce_process_on_gpu: Whether to enforce the augmentation to happen on the GPU, even if the input image is stored on the CPU.
                Default value is ``True``.
        '''

        self._image_name = image_name
        self._min_max_brightness = min_max_brightness
        self._min_max_hue = min_max_hue
        self._min_max_contrast = min_max_contrast
        self._min_max_saturation = min_max_saturation
        self._prob_brightness_aug = prob_brightness_aug
        self._prob_hue_aug = prob_hue_aug
        self._prob_contrast_aug = prob_contrast_aug
        self._prob_saturation_aug = prob_saturation_aug
        self._prob_swap_channels = prob_swap_channels
        self._enforce_process_on_gpu = enforce_process_on_gpu
        self._image_format = types.DALIImageType.BGR if is_bgr else types.DALIImageType.RGB

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # Note that only the actual image normalization becomes part of the DALI graph, the search for images as well as resolvong the paths obtained in order to access the image,
        # and the change of the data type, happen at graph construction time and do not influence the run time.

        # Find all images
        image_paths = data.find_all_occurrences(self._image_name)

        # Get the images, while remembering original dtypes. We will process in float
        # but return tensors with the original dtype.
        images = [None] * len(image_paths)
        image_types = [None] * len(image_paths)
        for i, ip in enumerate(image_paths):
            images[i] = data.get_item_in_path(ip)
            image_types[i] = data.get_type_of_item_in_path(ip)
            assert image_types[i] in (
                types.DALIDataType.FLOAT,
                types.DALIDataType.UINT8,
            ), f"Image type {image_types[i]} not supported"

        # Process the images
        self._process_images(images, image_types)

        # Set the updated images
        for i, ip in enumerate(image_paths):
            data.set_item_in_path(ip, images[i])

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        image_paths = data_empty.find_all_occurrences(self._image_name)

        # Make sure at least 1 image is available
        if len(image_paths) == 0:
            raise KeyError(
                f"No occurrences of images found. Fields containing images are expected to have the name '{self._image_name}', as specified in the constructor."
            )

        return data_empty

    def _process_images(self, images: Sequence[DataNode], image_types: Sequence[types.DALIDataType]):
        '''Process the images.'''

        augmentation = self._get_augmentation_setup()

        for i in range(len(images)):

            image_type = image_types[i]
            is_image_uint8 = image_type == types.DALIDataType.UINT8

            if self._enforce_process_on_gpu:
                images[i] = images[i].gpu()

            # Work in float domain in [0, 1] for all photometric ops
            images[i] = fn.cast(images[i], dtype=types.DALIDataType.FLOAT)
            if is_image_uint8:
                intensity_factor = 1.0 / 255.0
                images[i] = images[i] * intensity_factor
            else:
                intensity_factor = 1.0

            if augmentation["aug_brightness"]:
                images[i] = images[i] + augmentation["delta"] * intensity_factor
                images[i] = dali_math.clamp(images[i], 0.0, 1.0)
            if augmentation["contrast_mode"] == 1 and augmentation["aug_contrast"]:
                images[i] = images[i] * augmentation["alpha"]
                images[i] = dali_math.clamp(images[i], 0.0, 1.0)
            if augmentation["aug_saturation"]:
                images[i] = fn.saturation(
                    images[i],
                    saturation=augmentation["saturation"],
                    image_type=self._image_format,
                    dtype=types.DALIDataType.FLOAT,
                )
            if augmentation["aug_hue"]:
                images[i] = fn.hue(
                    images[i],
                    hue=augmentation["hue"],
                    image_type=self._image_format,
                    dtype=types.DALIDataType.FLOAT,
                )
            if augmentation["contrast_mode"] == 0 and augmentation["aug_contrast"]:
                images[i] = images[i] * augmentation["alpha"]
                images[i] = dali_math.clamp(images[i], 0.0, 1.0)
            if augmentation["aug_swap_channels"]:
                image = images[i]
                channel_permutation = augmentation["channel_permutation"]
                image_channels = [
                    image[:, :, channel_permutation[0]],
                    image[:, :, channel_permutation[1]],
                    image[:, :, channel_permutation[2]],
                ]
                images[i] = fn.stack(*image_channels, axis=2, axis_name='C')

            # Finally, cast back to original dtype and clamp to valid range
            if is_image_uint8:
                images[i] = images[i] * 255.0
                images[i] = dali_math.clamp(images[i], 0, 255)
                images[i] = fn.cast(images[i], dtype=image_type)
            else:
                images[i] = dali_math.clamp(images[i], 0.0, 1.0)

    def _get_augmentation_setup(self):

        def get_color_channel_permutation(perm_index: int) -> DataNode:
            # In total, there are 3! = 6 possibilities for permutations.
            # These are realized here as enumerated cases instead of a
            # permutation. This simplifies the processing and allows
            # for easier randomization (i.e. only need to generate a single
            # random number instead of sampling without replacement). The random
            # number is input to this function.
            res = fn.constant(idata=[0, 1, 2], shape=[3], dtype=types.DALIDataType.INT32)
            # Identity permutation
            if perm_index == 0:
                pass
            # Pair-wise switch elements (includes inverting the order)
            elif perm_index == 1:
                res = fn.constant(idata=[0, 2, 1], shape=[3], dtype=types.DALIDataType.INT32)
            elif perm_index == 2:
                res = fn.constant(idata=[1, 0, 2], shape=[3], dtype=types.DALIDataType.INT32)
            elif perm_index == 3:
                res = fn.constant(idata=[2, 1, 0], shape=[3], dtype=types.DALIDataType.INT32)
            # Shift elements by 1
            elif perm_index == 4:
                res = fn.constant(idata=[2, 0, 1], shape=[3], dtype=types.DALIDataType.INT32)
            elif perm_index == 5:
                res = fn.constant(idata=[1, 2, 0], shape=[3], dtype=types.DALIDataType.INT32)
            return res

        aug_brightness = (
            fn.random.uniform(range=[0.0, 1.0], dtype=types.DALIDataType.FLOAT) < self._prob_brightness_aug
        )
        aug_contrast = (
            fn.random.uniform(range=[0.0, 1.0], dtype=types.DALIDataType.FLOAT) < self._prob_contrast_aug
        )
        aug_saturation = (
            fn.random.uniform(range=[0.0, 1.0], dtype=types.DALIDataType.FLOAT) < self._prob_saturation_aug
        )
        aug_hue = fn.random.uniform(range=[0.0, 1.0], dtype=types.DALIDataType.FLOAT) < self._prob_hue_aug
        aug_swap_channels = (
            fn.random.uniform(range=[0.0, 1.0], dtype=types.DALIDataType.FLOAT) < self._prob_swap_channels
        )

        contrast_mode = fn.random.uniform(range=[0, 2], dtype=types.DALIDataType.INT8)

        delta = 0.0
        alpha = 0.0
        saturation = 0.0
        hue = 0.0
        channel_permutation = fn.constant(idata=[0, 1, 2], shape=[3], dtype=types.DALIDataType.INT32)

        if aug_brightness:
            delta = self._get_random_in_range(self._min_max_brightness)
        if aug_contrast:
            alpha = self._get_random_in_range(self._min_max_contrast)
        if aug_hue:
            hue = self._get_random_in_range(self._min_max_hue)
        if aug_saturation:
            saturation = self._get_random_in_range(self._min_max_saturation)
        if aug_swap_channels:
            pertmutation_index = fn.random.uniform(range=[0, 6], dtype=types.DALIDataType.INT32)
            channel_permutation = get_color_channel_permutation(pertmutation_index)

        res = {
            "contrast_mode": contrast_mode,
            "aug_brightness": aug_brightness,
            "aug_contrast": aug_contrast,
            "aug_saturation": aug_saturation,
            "aug_hue": aug_hue,
            "aug_swap_channels": aug_swap_channels,
            "delta": delta,
            "alpha": alpha,
            "hue": hue,
            "saturation": saturation,
            "channel_permutation": channel_permutation,
        }
        return res

    @staticmethod
    def _get_random_in_range(range):
        if range[1] == range[0]:
            res = range[0]
        else:
            res = fn.random.uniform(range=range, dtype=types.DALIDataType.FLOAT)
        return res
