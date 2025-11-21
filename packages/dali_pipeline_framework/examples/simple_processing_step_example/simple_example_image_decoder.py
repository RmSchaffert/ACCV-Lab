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

# @NOTE: Import the needed functinoality from the DALI pipeline framework package.
from accvlab.dali_pipeline_framework.processing_steps import PipelineStepBase
from accvlab.dali_pipeline_framework.pipeline import SampleDataGroup


# @NOTE: We derive from the base class PipelineStepBase to implement a processing step.
class SimpleExampleImageDecoder(PipelineStepBase):
    '''Minimal example: decode a single image.'''

    def __init__(self, use_device_mixed: bool, hw_decoder_load: float = 0.65):
        '''

        Args:
            use_device_mixed: If ``True``, decoding is performed on GPU (mixed) and results reside in GPU
                memory; if ``False``, only CPU is used.
            hw_decoder_load: When ``use_device_mixed==True``, fraction of work handled by hardware decoder
                (rest is handled by CUDA kernels).

        '''

        self._use_device_mixed = use_device_mixed
        self._hw_decoder_load = hw_decoder_load

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # @NOTE: This is the main processing step. This method needs to be implemented by the derived class.

        # @NOTE
        # Here, we assume that the image is stored in the data field "image" in the root of the input data.
        # This makes this processing step unflexible, as images anywhere else in the data cannot be processed
        # (and the processing is limited to a single image).
        image = data["image"]

        # @NOTE: Perform the actual decoding using the DALI image decoder operator.
        image = fn.decoders.image(
            image,
            device="mixed" if self._use_device_mixed else "cpu",
            hw_decoder_load=self._hw_decoder_load,
            output_type=types.RGB,
        )

        # @NOTE
        # Again, we assume a fixed image data field name and set the image data to the decoded version. Note
        # that this simple assignment works as the image is still of the same type (UINT8). If the type
        # changes, we would need to change the type `data` first, e.g. by calling:
        #  >> data.change_type_of_data_and_remove_data("image", new_type)
        # which would change the type and remove the reference to the old data (as the old data is by
        # definition incompatible with the new type). After this call, we could write `image` into the data
        # field.
        #
        # Similarly, if we wanted to add a new data fields, we would need to call:
        #  >> data.add_data_field("new_field", new_type)
        # to add the field to the data format before setting the actual data.
        data["image"] = image

        # @NOTE: Return the resulting data structure with the decoded image.
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        # @NOTE
        # Here, we need to override the base class method to adjust the data format to the output data
        # format & check for compatibility.

        # @NOTE
        # In this simple example processing step, the input data format is hardcoded (see above), and we just
        # check that the input data format contains an image field and that it is of the correct type.
        if not data_empty.has_child("image"):
            raise KeyError(f"Image data field not found")
        if data_empty.get_type_of_field("image") != types.DALIDataType.UINT8:
            raise ValueError(f"Image data field is not of type UINT8")

        # @NOTE
        # We do not need to add or adjust any new data fields, so we can return the input data format as is.
        # If we e.g. wanted to change the type of the image field, we would need to call:
        #  >> data_empty.change_type_of_data_and_remove_data("image", new_type)
        # before returning the adjusted blueprint. Note hat the output data structure needs to match the
        # structure of the actual output, so any changes to the data structure done here need to be reflected
        # in the _process() method as well.
        return data_empty
