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

from typing import Union, List, Tuple

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np

import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.types as types

from accvlab.dali_pipeline_framework.pipeline import SampleDataGroup

from accvlab.dali_pipeline_framework.processing_steps.pipeline_step_base import PipelineStepBase

from accvlab.dali_pipeline_framework.operators_impl import numba_operators as numba_op


class BEVFormerDataCombiner(PipelineStepBase):
    '''Prepare and combine inputs for BEVFormer training.

    Differences to the training format:
      - Individual fields
        - Some fields need to be converted (e.g., float32 → float64)
        - Bounding boxes are in the same format as in the training. However, the training uses a custom
          container type not available in DALI; conversion must happen outside the pipeline
        - Lidar-to-image transformation: Add a row to the transformation matrix ([0, 0, 0, 1])
      - Data format: Hierarchical structure differs from the training format, e.g.:
        - Images
          - Stored as a single tensor
          - Channel dimension precedes height and width
        - CAN-bus data
          - Stored as a single tensor
          - Expressed as relative values (vs. absolute) when configured

    Behavior:
    - Images: For each time step, per‑camera images are transposed to channel‑first (C,H,W), stacked across
      cameras (axis 'V'), then stacked across time (axis 'T') into ``images_time_step_cam``.
    - Camera geometry: For each time step and camera, a homogeneous bottom row ``[0, 0, 0, 1]`` is appended
      to ``lidar2img`` in place (resulting in 4×4 matrices).
    - CAN‑bus: Absolute or relative features (configurable) are concatenated into a single FLOAT tensor per
      time step. Provides orientation in radians normalized to ``[0, 2π]`` and an orientation value in
      degrees (absolute or relative between time steps).
    - Ground truth (optional): 3D boxes are combined into a FLOAT tensor (translations, sizes, orientation,
      velocity); NaNs are replaced with 0; the number of objects is stored as INT64.

    Fields:
    - Added (top level): ``images_time_step_cam`` (FLOAT)
    - Per time step: ``can_bus`` is kept and its dtype converted to FLOAT during format adjustment
    - Optional (if GT included): ``gt_boxes.bboxes`` (FLOAT), ``gt_boxes.num_bboxes`` (INT64)

    '''

    def __init__(
        self,
        compute_can_relative_values: bool,
        is_ground_truth_included: bool,
        bounding_box_offset_rel_box_size: Union[list, tuple] = [0.0, 0.0, -0.5],
    ):
        '''

        Args:
            compute_can_relative_values: Whether to compute relative CAN‑bus features (vs. absolute).
            is_ground_truth_included: Defines whether to also convert ground‑truth labels and boxes.
            bounding_box_offset_rel_box_size: Normalized offset added to box centers (per dimension),
                expressed relative to box size (e.g., ``0.5`` shifts by half the box size).
        '''

        self._compute_can_relative_values = compute_can_relative_values
        self._is_ground_truth_included = is_ground_truth_included
        self._bounding_box_offset_rel_box_size = bounding_box_offset_rel_box_size

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # ===== Combine all images =====
        # For each time step, transpose per-camera images to channel-first (C,H,W), stack over cameras
        # along axis 'V', then stack over time along axis 'T' to produce a single tensor
        # ``images_time_step_cam`` with layout [T, V, C, H, W].
        num_time_steps = len(data["data_at_time_steps"])
        num_cams = len(data["data_at_time_steps"][0]["cams"])

        per_time_step_image_blocks = []
        for t in range(num_time_steps):
            curr_timestep_data = data["data_at_time_steps"][t]
            curr_time_step_images = []
            for c in range(num_cams):
                curr_image = curr_timestep_data["cams"][c]["image"]
                curr_image_channels_outer = fn.transpose(curr_image, perm=[2, 0, 1])
                curr_time_step_images.append(curr_image_channels_outer)
            curr_time_step_image_block = fn.stack(*curr_time_step_images, axis=0, axis_name="V")
            per_time_step_image_blocks.append(curr_time_step_image_block)
        images = fn.stack(*per_time_step_image_blocks, axis=0, axis_name="T")

        # ===== Compute CAN-bus features =====
        # For each time step, gather CAN-bus signals and ego-pose, then concatenate features.
        # If ``compute_can_relative_values`` is True, compute translation and orientation deltas relative to
        # the previous time step (when ``prev_exists``); otherwise produce absolute quantities.
        # Orientation is provided both as radians in [0, 2π] and in degrees.
        per_time_step_can_data = [None] * num_time_steps
        prev_translation = 0
        prev_orientation = 0
        for t in range(num_time_steps):
            # Get the original data from CAN-bus and ego-pose
            curr_timestep_data = data["data_at_time_steps"][t]
            can_data = curr_timestep_data["can_bus"]
            ego_pose = curr_timestep_data["ego_pose"]
            # Get the individual data fields
            acceleration_in = can_data["acceleration"]
            rotation_rate_in = can_data["rotation_rate"]
            velocity_in = can_data["velocity"]
            rotation_in = ego_pose["rotation"]
            translation_in = ego_pose["translation"]
            orientation_in = ego_pose["orientation"]

            # Normalize orientation to [0, 2π]
            orientation_0_2pi = numba_op.ensure_range(
                orientation_in, 0.0, 2.0 * np.pi, 2.0 * np.pi, 0, types.DALIDataType.FLOAT
            )

            # ===== Compute relative CAN-bus features =====
            # Note that for the orientation, the relative orientation is computed for the value in
            # degrees. The value in radians is still provided as an absolute value. This is consistent
            # with the original training format.
            if self._compute_can_relative_values:
                # Relative translation and orientation (deg) if previous step exists; zeros otherwise
                if curr_timestep_data["prev_exists"]:
                    translation_rel = translation_in - prev_translation
                    # Wrap delta orientation to [-π, π] using numba op, then convert to degrees
                    orientation_delta = orientation_in - prev_orientation
                    orientation_delta_wrapped = numba_op.ensure_range(
                        orientation_delta, -np.pi, np.pi, 2.0 * np.pi, 0, types.DALIDataType.FLOAT
                    )
                    orientation_deg_rel = orientation_delta_wrapped * (180.0 / np.pi)
                else:
                    translation_rel = fn.zeros_like(translation_in)
                    orientation_deg_rel = fn.zeros_like(orientation_in)

                # Combine the data into a single tensor
                can_data_list = [
                    translation_rel,
                    rotation_in,
                    acceleration_in,
                    rotation_rate_in,
                    velocity_in,
                    fn.expand_dims(orientation_0_2pi, axes=0),
                    fn.expand_dims(orientation_deg_rel, axes=0),
                ]

                prev_translation = translation_in
                prev_orientation = orientation_in
            else:
                # Absolute quantities: include orientation in [0, 2π] and orientation in degrees
                orientation_deg = orientation_in * (180.0 / np.pi)
                can_data_list = [
                    translation_in,
                    rotation_in,
                    acceleration_in,
                    rotation_rate_in,
                    velocity_in,
                    fn.expand_dims(orientation_0_2pi, axes=0),
                    fn.expand_dims(orientation_deg, axes=0),
                ]

            # Combine the data into a single tensor ...
            can_data = fn.cat(*can_data_list, axis=0)
            # ... and store the combined data for the current time step
            per_time_step_can_data[t] = can_data

        # ===== Adjust lidar-to-image transformations =====
        # Append a homogeneous bottom row [0, 0, 0, 1] to each 3x4 matrix to make it 4x4.
        last_matrix_row = fn.constant(
            fdata=[0.0, 0.0, 0.0, 1.0], shape=[1, 4], dtype=types.DALIDataType.FLOAT
        )
        for t in range(num_time_steps):
            curr_timestep_data = data["data_at_time_steps"][t]
            for c in range(num_cams):
                curr_image_meta_data = curr_timestep_data["cams"][c]["image_meta"]
                curr_trafo = curr_image_meta_data["lidar2img"]
                curr_trafo_res = fn.cat(curr_trafo, last_matrix_row, axis=0)
                curr_image_meta_data["lidar2img"] = curr_trafo_res

        # ===== Prepare ground‑truth (if needed) =====
        # Read original GT (before format adjustment removes the source fields), convert to the training
        # layout, replace NaNs, and count objects. Actual storage happens after format adjustment below.
        if self._is_ground_truth_included:
            gt_boxes_data = data["gt_boxes"]

            translations = gt_boxes_data["translations"]
            sizes = gt_boxes_data["sizes"]
            orientations = gt_boxes_data["orientations"]
            velocities = gt_boxes_data["velocities"]

            # Adjust orientation to the expected reference frame and wrap to [-π, π].
            # This is done to be consistent with the original training format.
            orientations = -orientations - np.pi * 0.5
            # Shift the bounding boxes by the configured offset
            translations = translations + sizes * fn.constant(
                fdata=self._bounding_box_offset_rel_box_size, shape=[1, 3], dtype=types.DALIDataType.FLOAT
            )

            # Wrap the orientation to the expected range
            orientations = numba_op.ensure_range(
                orientations, -np.pi, np.pi, 2.0 * np.pi, 1, types.DALIDataType.FLOAT
            )

            # Combine the bounding box data into a single tensor
            bounding_boxes = fn.cat(
                translations, sizes, fn.expand_dims(orientations, axes=1), velocities[:, 0:2], axis=1
            )

            # Replace NaNs with 0
            bounding_boxes = numba_op.replace_nans(bounding_boxes, 0.0, 2)

            # Get the number of bounding boxes. Note that this is needed as the bounding box tensors are
            # padded to the same size, and we need to know the number of valid bounding boxes.
            num_bboxes = bounding_boxes.shape()[0]

        # ===== Adjust format and store results =====
        # Convert containers/types to the expected training layout, then store combined tensors.
        self._adjust_format(data)

        data["images_time_step_cam"] = images
        for t in range(num_time_steps):
            data["data_at_time_steps"][t]["can_bus"] = per_time_step_can_data[t]

        # Store the adjusted ground-truth bounding boxes
        if self._is_ground_truth_included:
            data["gt_boxes"]["bboxes"] = bounding_boxes
            data["gt_boxes"]["num_bboxes"] = num_bboxes

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        self._check_input_data_format(data_empty)
        self._adjust_format(data_empty)
        return data_empty

    def _adjust_format(self, data_inout: SampleDataGroup):
        data_inout.remove_all_occurrences("image")
        data_inout.add_data_field("images_time_step_cam", types.DALIDataType.FLOAT)
        can_bus_paths = data_inout.find_all_occurrences("can_bus")
        for cbp in can_bus_paths:
            parent = data_inout.get_parent_of_path(cbp)
            parent.remove_field("can_bus")
            parent.add_data_field("can_bus", types.DALIDataType.FLOAT)

        if self._is_ground_truth_included:
            gt_box_data = data_inout["gt_boxes"]
            gt_box_fields_to_remove = [
                "sizes",
                "rotations",
                "translations",
                "visibility_level",
                "orientations",
                "velocities",
            ]
            for to_remove in gt_box_fields_to_remove:
                gt_box_data.remove_field(to_remove)
            gt_box_data.add_data_field("bboxes", types.DALIDataType.FLOAT)
            gt_box_data.add_data_field("num_bboxes", types.DALIDataType.INT64)

    def _check_input_data_format(self, data: SampleDataGroup):
        if not data.has_child("data_at_time_steps"):
            raise ValueError("Input data ha no field `data_at_time_steps`")

        is_time_steps_data_group = data.is_data_group_field("data_at_time_steps")
        if not is_time_steps_data_group:
            raise ValueError("Input data field `data_at_time_steps` is not a data group field")

        time_steps = data["data_at_time_steps"]

        if not time_steps.is_array():
            raise ValueError(
                "The data group field `data_at_time_steps` is not organized as an array (see documentation of SampleDataGroup.is_array() for details)"
            )

        num_cams = None
        for ts in time_steps.contained_top_level_field_names:
            time_step = time_steps[ts]
            if self._compute_can_relative_values:
                if not time_step.has_child("prev_exists") or not time_step.is_data_field("prev_exists"):
                    raise ValueError(
                        f"Data at time step `{ts}` does not contain field `prev_exists` or the field is not a data field. This field is needed if `compute_can_relative_values` is set to `True` at construction."
                    )
            if not time_step.has_child("can_bus") or not time_step.is_data_group_field("can_bus"):
                raise ValueError(
                    f"Data at time step `{ts}` does not contain field `can_bus` or the field is not a data group field"
                )
            if not time_step.has_child("ego_pose") or not time_step.is_data_group_field("ego_pose"):
                raise ValueError(
                    f"Data at time step `{ts}` does not contain field `ego_pose` or the field is not a data group field"
                )
            if not time_step.has_child("cams") or not time_step.is_data_group_field("cams"):
                raise ValueError(
                    f"Data at time step `{ts}` does not contain field `cams` or the field is not a data group field"
                )

            can_bus_fields = ["acceleration", "rotation_rate", "velocity"]
            can_bus = time_step["can_bus"]
            for cbf in can_bus_fields:
                if not can_bus.has_child(cbf) or not can_bus.is_data_field(cbf):
                    raise ValueError(
                        f"Data at time step `{ts}` field `can_bus` does not contain `{cbf}, or `{cbf}` is not a data field"
                    )

            ego_pose_fields = ["rotation", "translation"]
            ego_pose = time_step["ego_pose"]
            for epf in ego_pose_fields:
                if not ego_pose.has_child(epf) or not ego_pose.is_data_field(epf):
                    raise ValueError(
                        f"Data at time step `{ts}` field `ego_pose` does not contain `{epf}`, or `{epf}` is not a data field"
                    )

            cams = time_step["cams"]
            if num_cams is None:
                num_cams = len(cams)

            if not cams.is_array():
                raise ValueError(
                    f"The data group field `cams` for time step {ts} is not organized as an array (see documentation of SampleDataGroup.is_array() for details)"
                )

            if num_cams != len(cams):
                raise ValueError(
                    f"Number of cameras for first time step ({num_cams}) and current time step {ts} ({len(cams)}) do not match"
                )

            for i in range(num_cams):
                cam = cams[i]
                if not cam.has_child("image") or not cam.is_data_field("image"):
                    raise ValueError(
                        f"Camera at time step{ts} and cam id {i} does not contain the field `image` or the field is not a data field"
                    )
                if not cam.has_child("image_meta") or not cam.is_data_group_field("image_meta"):
                    raise ValueError(
                        f"Camera at time step{ts} and cam id {i} does not contain the field `image_meta` or the field is not a data group field"
                    )
                image_meta = cam["image_meta"]
                if not image_meta.has_child("lidar2img") or not image_meta.is_data_field("lidar2img"):
                    raise ValueError(
                        f"Data group field `image_meta` for camera at time step {ts} and cam id {i} does not contain the field `lidar2img` or the field is not a data group field"
                    )

            if self._is_ground_truth_included:

                if not data.has_child("gt_boxes"):
                    raise ValueError("Input data ha no field `gt_boxes`")

                is_gt_boxes_data_group = data.is_data_group_field("gt_boxes")

                if not is_gt_boxes_data_group:
                    raise ValueError("Input data field `is_gt_boxes_data_group` is not a data group field")

                gt_boxes = data["gt_boxes"]

                gt_boxes_fields = ["categories", "sizes", "rotations", "translations", "visibility_level"]
                for gbf in gt_boxes_fields:
                    if not gt_boxes.has_child(gbf):
                        raise ValueError(
                            f"Data group field `gt_boxes` does not contain required field '{gbf}'"
                        )
