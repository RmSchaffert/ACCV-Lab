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

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import nvidia.dali.types as types
import nvidia.dali.fn as fn

# @NOTE: Import the needed functionality from the DALI pipeline framework package. Here, we are inside the
# `accvlab.dali_pipeline_framework` package, and so, in contrast to the simple step example, we use relative
# imports.
from .pipeline_step_base import PipelineStepBase
from ..pipeline.sample_data_group import SampleDataGroup


class ImageDecoder(PipelineStepBase):
    '''Decode images.

    Behavior:
      - Finds all images by name, decodes them (to RGB or BGR), and replaces the encoded image data
        by the decoded version in place.
      - Image search happens at DALI graph construction time; only the actual decoding operator is part of
        the DALI graph. This means that the runtime performance is not affected by the search for images.
    '''

    def __init__(
        self,
        image_name: str,
        use_device_mixed: bool,
        hw_decoder_load: float = 0.65,
        as_bgr: bool = False,
    ):
        '''

        Args:
            image_name: Name of the image data field(s) to decode
            use_device_mixed: If ``True``, decoding will be partially performed on the GPU and the resulting
                images will be located in GPU memory. If ``False``, only the CPU is used.
            hw_decoder_load: In case of ``use_device_mixed==True``, this parameter sets the fraction of the
                workload to be performed by decoding hardware (as opposed to software CUDA kernels).
            as_bgr: Whether to output BGR images (instead of RGB images).
        '''

        self._image_name = image_name
        self._use_device_mixed = use_device_mixed
        self._hw_decoder_load = hw_decoder_load
        self._as_bgr = as_bgr

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # @NOTE
        # We need to override the main processing step function and implement our functionality here.
        #
        # Note that functionality related to getting the image data fields (e.g.
        # `data.find_all_occurrences()`, `data.get_item_in_path()`, ...)
        # is performed at DALI graph construction time, and therefore is free in terms of runtime performance
        # when running the pipeline.

        # @NOTE
        # In contrast to the simple example (see
        # `packages/dali_pipeline_framework/examples/simple_processing_step_example/simple_example_image_decoder.py`),
        # we search for all images to process (same name, but may be in different locations in the input data,
        # e.g. one image per camera). Note that if more than one image name is needed, multiple `ImageDecoder`
        # steps can be used in the pipeline, one per name.
        image_paths = data.find_all_occurrences(self._image_name)

        # @NOTE
        # We loop over all images found. Note that while in the DALI graph, loops are not supported,
        # we can still use Python loops which do not depend on DALI-specific data, and are therefore executed
        # at graph construction time. Such loops are effectively unrolled in the DALI graph.

        # @NOTE: For each image found:
        for ip in image_paths:
            # @NOTE: Get the image to process. We use the `get_item_in_path()` method to get the image data at
            # the path `ip` without having to traverse the data structure manually.
            encoded_image = data.get_item_in_path(ip)

            # @NOTE
            # Decode the image (as in the simple example, but with the additional configuration parameter
            # `as_bgr` for improved flexibility)
            decoding_output_type = types.RGB if not self._as_bgr else types.BGR
            image = fn.decoders.image(
                encoded_image,
                device="mixed" if self._use_device_mixed else "cpu",
                hw_decoder_load=self._hw_decoder_load,
                output_type=decoding_output_type,
            )

            # @NOTE
            # Set the image data to the decoded version. Here, we use the `set_item_in_path()` method to set
            # the image data at the path `ip` without having to traverse the data structure manually.
            data.set_item_in_path(ip, image)

        # @NOTE: Return the resulting data structure with the decoded images.
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        # @NOTE
        # As in the simple example, we need to override and implement this function to adjust the data format
        # to the output data format & check for compatibility.

        # @NOTE
        # Here, we do not assume a specific image location (see notes in `_process()` above). Instead, we
        # enforce that at least one image can be found.
        image_paths = data_empty.find_all_occurrences(self._image_name)
        if len(image_paths) == 0:
            raise KeyError(
                f"No occurrences of images found. Fields containing images are expected to have the name "
                f"'{self._image_name}', as specified in the constructor."
            )

        # @NOTE: For each image found:
        for ip in image_paths:
            # @NOTE: We check the format of the image.
            image_parent = data_empty.get_parent_of_path(ip)
            if image_parent.get_type_of_field("image") != types.DALIDataType.UINT8:
                raise ValueError(f"Image data at path `{ip}` in the input data is not of type UINT8")

        # @NOTE
        # Return the output data format (which is the same as the input data format, as we did not add or
        # adjust any data fields).
        return data_empty
