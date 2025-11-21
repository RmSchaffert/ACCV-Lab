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

'''
This module contains the classes which represent individual processing steps as well as the respective
base classes, which can be used to implement custom processing steps (see :class:`PipelineStepBase`) as well
as access modifier wrapper steps (see :class:`GroupToApplyToSelectedStepBase`).

The individual processing steps are the building blocks of the pipeline, which is defined by a sequence of
processing steps (in addition to the input callable/iterable, see the ``inputs`` sub-module).
'''

# Base classes
from .pipeline_step_base import PipelineStepBase
from .group_to_apply_to_selected_step_base import GroupToApplyToSelectedStepBase

# Data group application steps
from .data_group_in_path_applied_step import DataGroupInPathAppliedStep
from .data_groups_with_name_applied_step import DataGroupsWithNameAppliedStep
from .data_group_array_in_path_elements_applied_step import DataGroupArrayInPathElementsAppliedStep
from .data_group_array_with_name_elements_applied_step import DataGroupArrayWithNameElementsAppliedStep

# Processing steps
from .image_decoder import ImageDecoder
from .image_to_tile_size_padder import ImageToTileSizePadder
from .image_range_01_normalizer import ImageRange01Normalizer
from .image_mean_std_dev_normalizer import ImageMeanStdDevNormalizer
from .photo_metric_distorter import PhotoMetricDistorter
from .affine_transformer import AffineTransformer
from .coordinate_cropper import CoordinateCropper
from .padding_to_uniform import PaddingToUniform
from .axes_layout_setter import AxesLayoutSetter
from .bounding_box_to_heatmap_converter import BoundingBoxToHeatmapConverter
from .annotation_element_condition_eval import AnnotationElementConditionEval
from .bev_bboxes_transformer_3d import BEVBBoxesTransformer3D

# Utility and filtering steps
from .visible_bbox_selector import VisibleBboxSelector
from .points_in_range_check import PointsInRangeCheck
from .conditional_element_removal import ConditionalElementRemover
from .unneeded_fields_remover import UnneededFieldRemover

__all__ = [
    # Base classes
    'PipelineStepBase',
    'GroupToApplyToSelectedStepBase',
    # Data group application steps
    'DataGroupInPathAppliedStep',
    'DataGroupsWithNameAppliedStep',
    'DataGroupArrayInPathElementsAppliedStep',
    'DataGroupArrayWithNameElementsAppliedStep',
    # Processing steps
    'ImageDecoder',
    'ImageToTileSizePadder',
    'ImageRange01Normalizer',
    'ImageMeanStdDevNormalizer',
    'PhotoMetricDistorter',
    'AffineTransformer',
    'CoordinateCropper',
    'PaddingToUniform',
    'AxesLayoutSetter',
    'BoundingBoxToHeatmapConverter',
    'AnnotationElementConditionEval',
    'BEVBBoxesTransformer3D',
    # Utility and filtering steps
    'VisibleBboxSelector',
    'PointsInRangeCheck',
    'ConditionalElementRemover',
    'UnneededFieldRemover',
]
