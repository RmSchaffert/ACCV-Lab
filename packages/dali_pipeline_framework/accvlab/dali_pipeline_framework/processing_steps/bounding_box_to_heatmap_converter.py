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

from typing import Union, Sequence, Optional, Tuple

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np

import nvidia.dali.fn as fn
import nvidia.dali.math as dmath
import nvidia.dali.types as types
import nvidia.dali.data_node as node

from ..pipeline.sample_data_group import SampleDataGroup
from .pipeline_step_base import PipelineStepBase

from ..operators_impl.python_operator_functions import (
    apply_clipping_and_get_with_clipping_info,
    get_is_active,
)
from ..operators_impl.numba_operators import get_center_from_bboxes, get_radii_from_bboxes


def _load_custom_operator():
    import os

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    custom_operator_file = os.path.join(parent_dir, "lib_draw_gaussians.so")
    import nvidia.dali.plugin_manager as plugin_manager

    plugin_manager.load_library(custom_operator_file, global_symbols=True)


try:
    _load_custom_operator()
except Exception as e:
    print(f"Warning: Failed to load custom operator: {e}")


class BoundingBoxToHeatmapConverter(PipelineStepBase):
    '''Convert 2D object bounding box annotations into Gaussian heatmaps.

    This step can process data from one or multiple cameras. It expects sibling fields in the input
    :class:`SampleDataGroup`: an image-size field and an annotation field containing bounding boxes (and
    optionally categories & bounding box centers). Multiple occurrences are supported; each is processed
    independently (see the constructor for details).

    Note:
        The input bounding boxes (and centers, if provided) are clipped to the image size and
        the corresponding output fields are corresponding to the clipped bounding boxes (scaled to the
        heatmap resolution).

    The following fields can be added inside each processed annotation. Note that apart from the heatmap,
    all fields are optional and can be omitted if not needed:

      - **heatmap**: Heatmap at the specified resolution. If per-category mode is enabled, the shape is
        ``[num_categories, H, W]``; otherwise ``[H, W]``. The data type is ``FLOAT``.
      - **is_active**: Boolean mask containing per-object flags indicating whether the object contributes
        to the heatmap (after clipping and threshold checks). Inactive objects were not drawn. Note that
        inactive objects are still contained in the other output fields.
      - **center**: Integer pixel center per object in heatmap coordinates (full-pixel location of the
        peak).
      - **center_offset**: Sub-pixel offset from the integer center to the true center in heatmap coordinates.
      - **height_width_bboxes_heatmap**: Per-object ``[height, width]`` in heatmap coordinates (after
        clipping and scaling from image to heatmap).
      - **bboxes_heatmap**: Per-object bounding box in heatmap coordinates (after clipping and scaling).

    To define the size of the individual Gaussians in the heatmap, the radius of the bounding boxes is used
    (with additional factors for the radius and the sigma-to-radius conversion of the Gaussians). The
    radius of the bounding boxes is defined as the distance between the center and the nearest edge of the
    bounding box. If the center is outside the box, the radius is 0 (and the minimum
    radius as defined on construction is enforced).
    '''

    def __init__(
        self,
        annotation_field_name: Union[str, int],
        bboxes_in_name: Union[str, int],
        heatmap_out_name: Union[str, int],
        heatmap_hw: Tuple[int, int],
        image_field_name: Optional[Union[str, int]] = None,
        image_hw_field_name: Optional[Union[str, int]] = None,
        categories_in_name: Optional[Union[str, int]] = None,
        num_categories: Optional[int] = None,
        min_object_size: Optional[Sequence[float]] = None,
        per_category_min_object_sizes: Optional[Sequence[Sequence[float]]] = None,
        use_per_category_heatmap: bool = True,
        is_valid_opt_in_name: Optional[Union[str, int]] = None,
        center_opt_in_name: Optional[Union[str, int]] = None,
        is_active_opt_out_name: Optional[Union[str, int]] = None,
        center_opt_out_name: Optional[Union[str, int]] = None,
        center_offset_opt_out_name: Optional[Union[str, int]] = None,
        height_width_bboxes_heatmap_opt_out_name: Optional[Union[str, int]] = None,
        bboxes_heatmap_opt_out_name: Optional[Union[str, int]] = None,
        min_fraction_area_clipping: float = 0.25,
        min_radius: float = 0.5,
        max_radius: float = 10.0,
        radius_scaling_factor: float = 0.8,
        radius_to_sigma_factor: float = 1.0 / 3.0,
    ):
        '''

        Args:
            annotation_field_name: Name of the field containing annotations. Bounding-box related fields
                are read from here and outputs are added here.
            bboxes_in_name: Name of the field containing bounding boxes.
            heatmap_out_name: Name of the output field to write the heatmap to.
            heatmap_hw: Heatmap size ``(height, width)``.
            image_field_name: Name of the field containing the image from which to extract the size.
                This field is expected to be a sibling field to the annotation field. Only one of
                ``image_field_name`` or ``image_hw_field_name`` should be set (single source of truth).
            image_hw_field_name: Name of the field containing the image height and width.
                This field is expected to be a sibling field to the annotation field. Only one of
                ``image_field_name`` or ``image_hw_field_name`` should be set (single source of truth).
            categories_in_name: Name of the field containing per-object categories. Required if any of the
                following holds: ``use_per_category_heatmap`` is ``True``,
                ``per_category_min_object_sizes`` is not ``None``, or ``num_categories`` is not ``None``.
                Otherwise set to ``None``.
            num_categories: Number of distinct categories. Objects with ``category >= num_categories`` are
                marked inactive. Set to ``None`` when categories are not used.
            min_object_size: Category-independent minimum object size ``[height, width]`` to be included.
                Must be ``None`` when ``per_category_min_object_sizes`` is not ``None``.
            per_category_min_object_sizes: Per-category minimum size ``[height, width]``. Must be ``None``
                when ``min_object_size`` is not ``None``.
            use_per_category_heatmap: If ``True``, draw a separate heatmap slice per category; otherwise draw
                a single heatmap.
            is_valid_opt_in_name: Optional field with per-object validity. Will be applied in addition to the
                internal checks to determine if an object is active. If absent, all objects are treated
                as valid (internal checks can still mark objects as inactive).
            center_opt_in_name: Name of the field containing the center of the bounding boxes.
                The so defined center is not necessarily the center of the 2D bounding box and could e.g.
                be the projection of the center of the 3D bounding box onto the image plane.
                Optional field. If not present, the centers are assumed to be the center of the 2D bounding
                boxes.
            is_active_opt_out_name: Output field name for the per-object active flag. Optional field. The
                corresponding field will not be added if not provided.
            center_opt_out_name: Output field name for integer center locations in the heatmap. The sub-pixel
                offset is written to ``center_offset_opt_out_name``. Optional field. The corresponding field
                will not be added if not provided.
            center_offset_opt_out_name: Output field name for sub-pixel center offsets in heatmap coordinates.
                Optional field. The corresponding field will not be added if not provided.
            height_width_bboxes_heatmap_opt_out_name: Output field for per-object ``[height, width]`` in the
                heatmap. Optional field. The corresponding field will not be added if not provided.
            bboxes_heatmap_opt_out_name: Output field for per-object bounding boxes in the heatmap.
                Optional field. The corresponding field will not be added if not provided.
            min_fraction_area_clipping: Minimum remaining area fraction after clipping for an object to be
                considered active. For example, with ``0.25``, boxes that lose more than ``75%`` of their
                area due to clipping are set inactive.
            min_radius: Minimum radius used when drawing Gaussians. Enforced lower bound is ``0.5``.
            max_radius: Maximum radius used when drawing Gaussians. Larger radii are clipped to this value.
            radius_scaling_factor: Scaling factor applied to the bbox-derived radius.
            radius_to_sigma_factor: Factor to convert radius to Gaussian sigma.
        '''

        # Validate constructor arguments according to the consitions as described in the docstring.

        # Ensure exactly one of image_field_name or image_hw_field_name is set (single source of truth)
        if (image_field_name is None) == (image_hw_field_name is None):
            raise ValueError(
                "Exactly one of 'image_field_name' or 'image_hw_field_name' must be set (single source of truth for image size)."
            )

        # categories_in_name is required when per-category heatmaps are used or when category-related
        # parameters are provided (as per docstring). Our current implementation always uses categories.
        categories_required = (
            use_per_category_heatmap
            or num_categories is not None
            or per_category_min_object_sizes is not None
        )

        if categories_required:
            assert (
                categories_in_name is not None
            ), "categories_in_name must be provided if categories are used (i.e. when use_per_category_heatmap=True, num_categories is not None, or per_category_min_object_sizes is not None)."
            # num_categories is required for per-category heatmaps (as per docstring) and must be positive.
            assert num_categories > 0, "num_categories must be a positive integer (if used)."

        # min_object_size and per_category_min_object_sizes are mutually exclusive (as per docstring).
        assert not (
            (min_object_size is not None) and (per_category_min_object_sizes is not None)
        ), "min_object_size and per_category_min_object_sizes are mutually exclusive; provide only one or None."

        # If per-category size thresholds are provided, validate their shape matches num_categories and [h, w].
        if per_category_min_object_sizes is not None:
            assert (
                len(per_category_min_object_sizes) == num_categories
            ), "per_category_min_object_sizes must have length equal to num_categories."
            for size_pair in per_category_min_object_sizes:
                assert (
                    isinstance(size_pair, (list, tuple)) and len(size_pair) == 2
                ), "Each entry in per_category_min_object_sizes must be a [height, width] pair."

        # Basic sanity checks for heatmap size.
        assert (
            isinstance(heatmap_hw, (list, tuple)) and len(heatmap_hw) == 2
        ), "heatmap_hw must be a (height, width) pair."
        assert heatmap_hw[0] > 0 and heatmap_hw[1] > 0, "heatmap_hw dimensions must be positive."

        # Parameters for the heatmap generation
        self._image_field_name = image_field_name
        self._image_hw_field_name = image_hw_field_name
        self._extract_size_from_image = image_field_name is not None
        self._annotation_field_name = annotation_field_name
        self._heatmap_hw = heatmap_hw
        self._num_categories = num_categories
        self._min_object_size = min_object_size
        self._use_per_category_heatmap = use_per_category_heatmap
        self._min_fraction_area_clipping = min_fraction_area_clipping
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._radius_scaling_factor = radius_scaling_factor
        self._radius_to_sigma_factor = radius_to_sigma_factor
        if per_category_min_object_sizes is not None:
            self._per_class_min_object_size_thresholds = np.array(per_category_min_object_sizes)
        else:
            self._per_class_min_object_size_thresholds = None

        # Name of the inputs & outputs
        self._bboxes_name = bboxes_in_name
        self._categories_name = categories_in_name
        self._is_valid_name = is_valid_opt_in_name
        self._center_in_name = center_opt_in_name
        self._heatmap_name = heatmap_out_name
        self._is_active_name = is_active_opt_out_name
        self._center_out_name = center_opt_out_name
        self._center_offset_name = center_offset_opt_out_name
        self._height_width_bboxes_heatmap_name = height_width_bboxes_heatmap_opt_out_name
        self._bboxes_heatmap_name = bboxes_heatmap_opt_out_name

        # Helper flags based on the inputs
        self._check_sizes = min_object_size is not None or per_category_min_object_sizes is not None
        self._use_per_category_size_check = per_category_min_object_sizes is not None
        self._check_categories = num_categories is not None

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        annotation_paths = data.find_all_occurrences(self._annotation_field_name)

        for ap in annotation_paths:
            parent = data.get_parent_of_path(ap)

            if self._extract_size_from_image:
                # Extract size from image using .shape() method
                image = parent[self._image_field_name]
                image_shape = image.shape()
                # Use fn.stack to create a proper tensor with [height, width]
                # Cast to int32 for consistency with heatmap operations
                image_hw = fn.cast(fn.stack(image_shape[-3], image_shape[-2]), dtype=types.DALIDataType.INT32)
            else:
                # Use size field
                image_hw = parent[self._image_hw_field_name]

            annotation = parent[self._annotation_field_name]

            annotation, image_hw = self._generate_heat_map(annotation, image_hw)

            # Only update image_hw field if we're using size fields (not extracting from images)
            if not self._extract_size_from_image:
                parent[self._image_hw_field_name] = image_hw
            parent[self._annotation_field_name] = annotation

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        annotation_paths = data_empty.find_all_occurrences(self._annotation_field_name)
        if len(annotation_paths) == 0:
            raise annotation_paths(
                f"No occurrences of images found. Fields containing images are expected to have the name '{self._image_field_name}', as specified in the constructor."
            )
        for ap in annotation_paths:
            parent = data_empty.get_parent_of_path(ap)

            if self._extract_size_from_image:
                # Check for image field
                if not self._image_field_name in parent.contained_top_level_field_names:
                    raise KeyError(
                        f"For annotation in path '{ap}', no image was found (i.e. no sibling field with name '{self._image_field_name}')"
                    )
            else:
                # Check for size field
                if not self._image_hw_field_name in parent.contained_top_level_field_names:
                    raise KeyError(
                        f"For annotation in path '{ap}', no image size information was found (i.e. no sibling field with name '{self._image_hw_field_name}')"
                    )
            if not self._bboxes_name in parent[self._annotation_field_name].contained_top_level_field_names:
                raise KeyError(
                    f"Inside annotation in path '{ap}', no '{self._bboxes_name}' fields was not found."
                )
            if self._center_in_name is not None:
                if (
                    not self._center_in_name
                    in parent[self._annotation_field_name].contained_top_level_field_names
                ):
                    raise KeyError(
                        f"Inside annotation in path '{ap}', no '{self._center_in_name}' fields was not found."
                    )

            annotations = parent[self._annotation_field_name]

            self._add_fields_to_annotations(annotations)

        return data_empty

    def _generate_heat_map(self, annotations: SampleDataGroup, image_hw: node.DataNode):

        # Get an empty heatmap image. Keep the leading class dimension for draw_gaussians; use 1 when not per-category.
        num_slices = self._num_categories if self._use_per_category_heatmap else 1
        heat_map = fn.constant(
            fdata=0.0,
            shape=[num_slices, self._heatmap_hw[0], self._heatmap_hw[1]],
            dtype=types.DALIDataType.FLOAT,
            device="cpu",
        )

        # Scaling between the image and the heat map
        scaling_transformation = fn.transforms.scale(
            scale=fn.stack(self._heatmap_hw[1] / image_hw[1], self._heatmap_hw[0] / image_hw[0])
        )

        categories = annotations[self._categories_name] if self._categories_name is not None else None
        bboxes = annotations[self._bboxes_name]

        if self._center_in_name is not None:
            center_in = annotations[self._center_in_name]
        else:
            center_in = get_center_from_bboxes(
                bboxes, dtype_in=annotations.get_type_of_field(self._bboxes_name)
            )

        # Perform clipping of bounding boxes & additional operations such as getting the resulting fraction of resulting & original bbox area and center of croppped bounding box.
        # Operations are performed in one operator to not introduce additional overhead
        bboxes_clipped, centers_clipped, hw_clipped, fraction_areas = fn.python_function(
            bboxes,
            center_in,
            scaling_transformation,
            self._heatmap_hw,
            num_outputs=4,
            function=apply_clipping_and_get_with_clipping_info,
        )

        # Get a full pixel location, as we want the peak to be in the center of the pixel (to avoid sub-pixel maximum detection later). Therefore,
        # get the center location pixel. Note that using 'floor' means that, e.g., values in [0, 1) belong to the first pixel. This means we assume
        # the coordinates (0, 0) to be the upper-left border of the upper-left pixel.
        center_full_pixel = fn.cast(dmath.floor(centers_clipped), dtype=types.DALIDataType.INT32)

        # This is the offset of the pixel
        center_offset = centers_clipped - center_full_pixel

        # Note the use of the bitwise "&" operator. Logical operators cannot be used in DALI for tensors with more than one element, and bitwise operators are equivalent for bool values.
        is_active = self._get_is_active(
            hw_clipped,
            (
                categories
                if (
                    self._use_per_category_heatmap
                    or self._check_categories
                    or self._use_per_category_size_check
                )
                else None
            ),
            fraction_areas,
        )
        # 'is_valid' is optional. Therefore, it is not listed in '_get_needed_fields_and_types()'. If it is not available, assume all objects are valid
        if self._is_valid_name in annotations.contained_top_level_field_names:
            is_active = is_active & annotations[self._is_valid_name]

        radii = get_radii_from_bboxes(
            bboxes_clipped, centers=centers_clipped, scaling_factor=self._radius_scaling_factor
        )
        radii = dmath.min(dmath.max(self._min_radius, radii), self._max_radius)

        # Example of how to print tensors during graph runtime:
        # print_tensor_op(radii, "radii")

        # Slice IDs: category IDs for per-category heatmap, 0 otherwise (single slice)
        slice_ids = (
            categories
            if self._use_per_category_heatmap
            else fn.cast(fraction_areas * 0, dtype=types.DALIDataType.INT32)
        )

        heat_map = self._draw_gaussians(heat_map, is_active, slice_ids, center_full_pixel, radii, num_slices)

        # Squeeze class dimension for single-heatmap mode
        if not self._use_per_category_heatmap:
            heat_map = fn.reshape(heat_map, shape=[self._heatmap_hw[0], self._heatmap_hw[1]])

        self._add_fields_to_annotations(annotations)

        annotations[self._heatmap_name] = heat_map
        if self._is_active_name is not None:
            annotations[self._is_active_name] = is_active
        if self._center_out_name is not None:
            annotations[self._center_out_name] = center_full_pixel
        if self._center_offset_name is not None:
            annotations[self._center_offset_name] = center_offset
        if self._height_width_bboxes_heatmap_name is not None:
            annotations[self._height_width_bboxes_heatmap_name] = hw_clipped
        if self._bboxes_heatmap_name is not None:
            annotations[self._bboxes_heatmap_name] = bboxes_clipped

        return annotations, image_hw

    def _add_fields_to_annotations(self, annotations: SampleDataGroup):
        try:
            annotations.add_data_field(self._heatmap_name, types.DALIDataType.FLOAT)
        except KeyError as e:
            raise KeyError(
                f"The input annotation must not contain the field '{self._heatmap_name}', as it is added by this step (name configurable on construction)."
            ) from e
        if self._is_active_name is not None:
            try:
                annotations.add_data_field(self._is_active_name, types.DALIDataType.BOOL)
            except KeyError as e:
                raise KeyError(
                    f"The input annotation must not contain the field '{self._is_active_name}', as it is added by this step (name configurable on construction)."
                ) from e
        if self._center_out_name is not None:
            try:
                annotations.add_data_field(self._center_out_name, types.DALIDataType.INT32)
            except KeyError as e:
                raise KeyError(
                    f"The input annotation must not contain the field '{self._center_out_name}', as it is added by this step (name configurable on construction)."
                ) from e
        if self._center_offset_name is not None:
            try:
                annotations.add_data_field(self._center_offset_name, types.DALIDataType.FLOAT)
            except KeyError as e:
                raise KeyError(
                    f"The input annotation must not contain the field '{self._center_offset_name}', as it is added by this step (name configurable on construction)."
                ) from e
        if self._height_width_bboxes_heatmap_name is not None:
            try:
                annotations.add_data_field(self._height_width_bboxes_heatmap_name, types.DALIDataType.FLOAT)
            except KeyError as e:
                raise KeyError(
                    f"The input annotation must not contain the field '{self._height_width_bboxes_heatmap_name}', as it is added by this step (name configurable on construction)."
                ) from e
        if self._bboxes_heatmap_name is not None:
            try:
                annotations.add_data_field(self._bboxes_heatmap_name, types.DALIDataType.FLOAT)
            except KeyError as e:
                raise KeyError(
                    f"The input annotation must not contain the field '{self._bboxes_heatmap_name}', as it is added by this step (name configurable on construction)."
                ) from e

    def _get_is_active(
        self,
        hw_clip: node.DataNode,
        classes: Optional[node.DataNode],
        fraction_areas: node.DataNode,
    ):
        if classes is None:
            min_obj_size_to_set = (
                np.array(self._min_object_size) if self._min_object_size is not None else None
            )
            function_to_use = lambda hw_clip, fraction_areas: get_is_active(
                hw_clip,
                classes=None,
                fraction_areas=fraction_areas,
                min_object_size=min_obj_size_to_set,
                per_class_min_object_sizes=None,
                num_classes=None,
                min_fraction_area_thresh=self._min_fraction_area_clipping,
            )
            active = fn.python_function(
                hw_clip,
                fraction_areas,
                function=function_to_use,
            )
        else:
            per_class_min_obj_size_to_set = (
                np.array(self._per_class_min_object_size_thresholds)
                if self._per_class_min_object_size_thresholds is not None
                else None
            )
            function_to_use = lambda hw_clip, classes, fraction_areas: get_is_active(
                hw_clip,
                classes,
                fraction_areas,
                min_object_size=None,
                per_class_min_object_sizes=per_class_min_obj_size_to_set,
                num_classes=self._num_categories,
                min_fraction_area_thresh=self._min_fraction_area_clipping,
            )
            active = fn.python_function(
                hw_clip,
                classes,
                fraction_areas,
                function=function_to_use,
            )
        return active

    def _draw_gaussians(self, heatmap, active, slice_ids, centers, radii, num_slices):
        # Use the custom operator for drawing Gaussians.
        heatmap = fn.draw_gaussians(
            heatmap,
            fn.cast(active, dtype=types.DALIDataType.BOOL),
            slice_ids,
            centers,
            radii,
            k_for_classes=[1.0] * num_slices,
            radius_to_sigma_factor=self._radius_to_sigma_factor,
        )
        return heatmap
