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

import pytest

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import PaddingToUniform


class _PadUniformProvider(DataProvider):
    """Provider with only non-scalar array fields and variable shapes per sample."""

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Vary height and 1D lengths by sample
        heights = [32, 48, 64]
        height = heights[sample_id % len(heights)]

        # Image HxWxC
        res["image"] = np.random.randint(0, 256, (height, 64, 3), dtype=np.uint8)

        # 1D float arrays
        lengths = [3, 4, 5]
        length = lengths[sample_id % len(lengths)]
        res["metadata"]["values"] = np.linspace(0.0, 1.0, num=length, dtype=np.float32)

        nested_lengths = [2, 3, 4]
        nested_length = nested_lengths[sample_id % len(nested_lengths)]
        res["metadata"]["features"] = np.linspace(10.0, 20.0, num=nested_length, dtype=np.float32)

        return res

    @override
    def get_number_of_samples(self) -> int:
        return 12

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()
        res.add_data_field("image", DALIDataType.UINT8)

        metadata = SampleDataGroup()
        metadata.add_data_field("values", DALIDataType.FLOAT)
        metadata.add_data_field("features", DALIDataType.FLOAT)
        res.add_data_group_field("metadata", metadata)
        return res


def test_pad_to_uniform_all_fields_in_pipeline():
    provider = _PadUniformProvider()
    input_callable = ShuffledShardedInputCallable(
        data_provider=provider,
        batch_size=3,
        shard_id=0,
        num_shards=1,
        shuffle=False,
    )

    # Pad all contained fields to uniform shapes
    step = PaddingToUniform(field_names=None, fill_value=0.0)

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=3,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(3, pipeline, pipeline_def.check_and_get_output_data_structure())
    batch = next(iter(iterator))

    # Validate shapes are uniform across the batch for all fields
    # No scalars should be present by construction
    assert batch["image"].shape[1:] == (64, 64, 3)
    assert batch["metadata"]["values"].shape[1:] == (5,)  # max of [3,4,5]
    assert batch["metadata"]["features"].shape[1:] == (4,)  # max of [2,3,4]


if __name__ == "__main__":
    pytest.main([__file__])
