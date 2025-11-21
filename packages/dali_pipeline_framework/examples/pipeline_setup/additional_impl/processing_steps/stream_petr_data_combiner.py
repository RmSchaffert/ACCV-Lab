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
Here, we define a processing step that is used to convert the format of the data to a format close to the
format expected by the StreamPETR training.

Note that the final format cannot be exactly matched, as it contains some framework-specific data types, which
are not supported by DALI.

However, it is advisable to bring the format as close as feasible to the final format within the DALI
pipeline. While a post-processing step can be used to bridge the remaining gap, processing within the DALI
pipeline is preferable for performance reasons. The post-processing step runs in the main training thread,
and therefore, it should be kept as light-weight as possible, ideally only containing type conversions or
reshaping operations, no actual data processing.


'''

from typing import Union

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import functools

import numpy as np

import nvidia.dali.fn as fn
import nvidia.dali.types as types

from accvlab.dali_pipeline_framework.processing_steps.pipeline_step_base import PipelineStepBase
from accvlab.dali_pipeline_framework.pipeline import SampleDataGroup

from accvlab.dali_pipeline_framework.operators_impl import numba_operators as numba_op
from accvlab.dali_pipeline_framework.operators_impl import python_operator_functions
from accvlab.dali_pipeline_framework.internal_helpers import debug_helpers as dbh


class StreamPETRDataCombiner(PipelineStepBase):
    '''Prepare and combine inputs for StreamPETR training.

    Some framework‑specific containers and batching conventions cannot be represented in DALI, so a small
    conversion outside the pipeline is still required. This step brings the data as close as feasible to the
    final format within the DALI graph.

    Behavior:

      - Images: Per‑camera images are transposed to channel‑first (C,H,W) and stacked over cameras into
        ``img``.
      - Camera geometry: Per‑camera ``lidar2img`` matrices get a homogeneous bottom row appended;
        ``intrinsics`` are padded to 4×4 by appending a zero column and a homogeneous bottom row;
        ``extrinsics`` are assumed to be 4×4 and are stacked as‑is. All are stacked across cameras into
        ``lidar2img``, ``intrinsics``, and ``extrinsics``.
      - Timestamps: Per‑camera timestamps are stacked into ``img_timestamp``.
      - Ego pose: ``ego_pose_geom.lidar_ego_pose`` is copied to ``ego_pose`` and ``lidar_ego_pose_inv`` to
        ``ego_pose_inv``.
      - Ground truth (optional): 3D boxes and labels are converted to training tensors; NaNs are replaced
        with 0; valid object counts per view are produced (as the corresponding tensors are padded, and the
        number of valid entries needs to be known).
        2D per‑camera ground truth is padded to common sizes then stacked.

    Fields:
      - Added: ``img``, ``lidar2img``, ``intrinsics``, ``extrinsics``, ``ego_pose``, ``ego_pose_inv``,
        ``img_timestamp``. ``num_gt_objects_3d`` is also added for compatibility. If GT is included, the
        following are populated: ``gt_bboxes_3d``, ``gt_labels_3d``, ``num_gt_objects_3d`` (count),
        and the stacked per‑camera tensors ``gt_bboxes``, ``gt_labels``, ``depths``, ``centers2d``,
        ``num_gt_objects``.
      - Removed after conversion: ``cams``, ``gt_boxes``, ``ego_pose_geom``.
    '''

    def __init__(
        self,
        is_ground_truth_included,
        bounding_box_offset_rel_box_size: Union[list, tuple] = [0.0, 0.0, -0.5],
    ):

        self._is_ground_truth_included = is_ground_truth_included
        self._bounding_box_offset_rel_box_size = bounding_box_offset_rel_box_size

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # ===== Add and initialize output fields =====
        # @NOTE Add new fields where the converted data will be stored. Original fields remain available
        # during conversion to allow referencing them while populating the new fields.
        # Some fields will be direct references; others are converted/stacked first.
        self._add_new_fields(data)

        # ===== Combine camera images =====
        # @NOTE Transpose per‑camera images to channel‑first (C,H,W), then stack across cameras into one tensor
        # with camera axis first (axis_name 'V').
        num_cams = len(data["cams"])

        images = []
        for c in range(num_cams):
            curr_image = data["cams"][c]["image"]
            curr_image_channels_outer = fn.transpose(curr_image, perm=[2, 0, 1])
            images.append(curr_image_channels_outer)
        images_block = fn.stack(*images, axis=0, axis_name="V")

        # ===== Adjust and combine projection geometry =====
        # @NOTE Pad per‑camera matrices to homogeneous 4×4 where needed (intrinsics), and append homogeneous
        # bottom row to lidar2img. Then stack across cameras.

        # Define constants used to pad matrices to homogeneous 4×4 form
        last_matrix_row = fn.constant(
            fdata=[0.0, 0.0, 0.0, 1.0], shape=[1, 4], dtype=types.DALIDataType.FLOAT
        )
        col_zeros_3 = fn.constant(fdata=[0.0, 0.0, 0.0], shape=[3, 1], dtype=types.DALIDataType.FLOAT)

        # Combine lidar2img, extrinsics, and intrinsics (for now as a list) & add padding as needed
        all_lidar2img = [None] * num_cams
        all_extr = [None] * num_cams
        all_intr = [None] * num_cams
        for c in range(num_cams):
            curr_cam_geometry = data["cams"][c]["cam_geometry"]
            curr_lidar2img = curr_cam_geometry["lidar2img"]
            curr_extr = curr_cam_geometry["extr_lidar2img"]
            curr_intr = curr_cam_geometry["intr_lidar2img"]

            # Append a homogeneous bottom row to lidar2img
            curr_lidar2img = fn.cat(curr_lidar2img, last_matrix_row, axis=0)

            # Pad intrinsics to 4×4 by adding a zero column and a homogeneous bottom row
            curr_intr = fn.cat(curr_intr, col_zeros_3, axis=1)
            curr_intr = fn.cat(curr_intr, last_matrix_row, axis=0)

            all_lidar2img[c] = curr_lidar2img
            all_extr[c] = curr_extr
            all_intr[c] = curr_intr

        # Combine the data from all cameras into single tensors (stack across cameras)
        lidar2img_block = fn.stack(*all_lidar2img, axis=0)
        extrinsics_block = fn.stack(*all_extr, axis=0)
        intrinsics_block = fn.stack(*all_intr, axis=0)

        # Combine camera timestamps into a single tensor (stack across cameras)
        timestamps = [None] * num_cams
        for c in range(num_cams):
            timestamps[c] = data["cams"][c]["timestamp"]
        timestamps_block = fn.stack(*timestamps, axis=0)

        # ===== Store combined tensors =====
        # @NOTE Store combined images, projection geometry, ego poses, and timestamps in the newly added
        # fields.
        data["img"] = images_block
        data["lidar2img"] = lidar2img_block
        data["intrinsics"] = intrinsics_block
        data["extrinsics"] = extrinsics_block
        data["ego_pose"] = data["ego_pose_geom"]["lidar_ego_pose"]
        data["ego_pose_inv"] = data["ego_pose_geom"]["lidar_ego_pose_inv"]
        data["img_timestamp"] = timestamps_block

        # ===== Process ground truth =====
        # @NOTE If ground truth is included, convert 3D boxes to the training layout and stack per‑camera 2D
        # GT. Also, replace NaNs, wrap angles to `[−pi, pi]`.
        #
        # Note that to combine the per-camera 2D GT, we need to pad the data to a common size. In order
        # to know the number of valid objects per camera after padding, we additionally store the number of
        # objects (i.e. the size of the data before padding).
        if self._is_ground_truth_included:
            # Adjust ground-truth bounding boxes.
            gt_boxes_data = data["gt_boxes"]
            sizes = gt_boxes_data["sizes"]
            # Add the translation offset to the translations
            translations = gt_boxes_data["translations"]
            translations = translations + sizes * fn.constant(
                fdata=self._bounding_box_offset_rel_box_size, shape=[1, 3], dtype=types.DALIDataType.FLOAT
            )

            # Reorder the size dimensions to the expected format
            sizes_reordered = fn.stack(sizes[:, 1], sizes[:, 0], sizes[:, 2], axis=1)

            # Get the orientation and velocity
            orientations = gt_boxes_data["orientations"]
            velocities = gt_boxes_data["velocities"]

            # Wrap orientation to the expected range [-π, π]
            orientations = numba_op.ensure_range(
                orientations, -np.pi, np.pi, 2.0 * np.pi, 1, types.DALIDataType.FLOAT
            )
            # Combine the bounding box data into a single tensor
            bounding_boxes = fn.cat(
                translations,
                sizes_reordered,
                fn.expand_dims(orientations, axes=1),
                velocities[:, 0:2],
                axis=1,
            )
            # Replace NaNs with 0
            bounding_boxes = numba_op.replace_nans(bounding_boxes, 0.0, 2)
            # Get the number of objects (needed because tensors are padded to common sizes)
            num_bboxes = bounding_boxes.shape()[0]

            # Store results
            data["gt_bboxes_3d"] = bounding_boxes
            data["gt_labels_3d"] = data["gt_boxes"]["categories"]
            data["num_gt_objects_3d"] = num_bboxes

            # ----- Process 2D image ground truth data -----
            # @NOTE Combine per‑camera 2D GT and record per‑camera counts before padding to a common size.
            bboxes = [data["cams"][c]["gt_boxes_2d"]["bboxes"] for c in range(num_cams)]
            gt_labels = [data["cams"][c]["gt_boxes_2d"]["categories"] for c in range(num_cams)]
            depths = [data["cams"][c]["gt_boxes_2d"]["depths"] for c in range(num_cams)]
            centers2d = [data["cams"][c]["gt_boxes_2d"]["centers"] for c in range(num_cams)]
            num_objects = [d.shape()[0] for d in depths]

            # Pad the data to a common size
            # @NOTE
            # Here, we use the `pad_to_common_size` function from the `python_operator_functions` module.
            # Please refer to the additional API reference for more details.
            pad_function = functools.partial(python_operator_functions.pad_to_common_size, fill_value=0.0)
            bboxes = fn.python_function(*bboxes, function=pad_function, num_outputs=num_cams)
            gt_labels = fn.python_function(*gt_labels, function=pad_function, num_outputs=num_cams)
            depths = fn.python_function(*depths, function=pad_function, num_outputs=num_cams)
            centers2d = fn.python_function(*centers2d, function=pad_function, num_outputs=num_cams)

            # Combine padded data into single tensors & add the number of objects per cameras
            data["gt_bboxes"] = fn.stack(*bboxes, axis=0)
            data["gt_labels"] = fn.stack(*gt_labels, axis=0)
            data["depths"] = fn.stack(*depths, axis=0)
            data["centers2d"] = fn.stack(*centers2d, axis=0)
            data["num_gt_objects"] = fn.stack(*num_objects, axis=0)

        # ===== Convert `prev_exists` to float =====
        # @NOTE
        # Here, we convert the `prev_exists` data field to `float`. This is needed because the training
        # expects a float data type for this field. Note that the type of the data field needs to be changed
        # explicitly using `change_type_of_data_and_remove_data`, before the converted data can be set.
        prev_exists = data["prev_exists"]
        prev_exists_fl = fn.cast(prev_exists, dtype=types.DALIDataType.FLOAT)
        data.change_type_of_data_and_remove_data("prev_exists", types.DALIDataType.FLOAT)
        data["prev_exists"] = prev_exists_fl

        # ===== Cleanup =====
        # @NOTE Remove fields that are no longer needed once combined tensors have been produced.
        self._remove_unneeded_fields(data)

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        self._check_input_data_format(data_empty)
        self._adjust_format(data_empty)
        data_empty.change_type_of_data_and_remove_data("prev_exists", types.DALIDataType.FLOAT)
        return data_empty

    # ===== Data structure checks and & output adjustments =====
    # @NOTE These helpers validate the input schema and add/remove fields for the output schema. See also
    # the 2D object detection pipeline for a broader discussion of output blueprints.

    def _adjust_format(self, data_inout: SampleDataGroup):
        # Add new data fields and delete old fields.
        self._add_new_fields(data_inout)
        self._remove_unneeded_fields(data_inout)

    def _add_new_fields(self, data_inout: SampleDataGroup):
        num_cams = len(data_inout["cams"])
        if self._is_ground_truth_included:
            fields_to_add_float = [
                "img",
                "lidar2img",
                "intrinsics",
                "extrinsics",
                "ego_pose",
                "ego_pose_inv",
                "gt_bboxes_3d",
                "gt_bboxes",
                "depths",
                "centers2d",
            ]
            fields_to_add_double = ["img_timestamp"]
            fields_to_add_long = ["gt_labels_3d", "num_gt_objects_3d", "gt_labels", "num_gt_objects"]
        else:
            fields_to_add_float = ["img", "lidar2img", "intrinsics", "extrinsics", "ego_pose", "ego_pose_inv"]
            fields_to_add_double = ["img_timestamp"]
            fields_to_add_long = []

        for fnm in fields_to_add_float:
            data_inout.add_data_field(fnm, types.DALIDataType.FLOAT)
        for fnm in fields_to_add_double:
            data_inout.add_data_field(fnm, types.DALIDataType.FLOAT64)
        for fnm in fields_to_add_long:
            data_inout.add_data_field(fnm, types.DALIDataType.INT64)

    def _remove_unneeded_fields(self, data_inout: SampleDataGroup):
        data_inout.remove_field("cams")
        data_inout.remove_field("gt_boxes")
        data_inout.remove_field("ego_pose_geom")

    def _check_input_data_format(self, data: SampleDataGroup):
        # Top-level fields
        data.check_has_children(
            data_field_children=["timestamp", "prev_exists"],
            data_group_field_children=(
                ["ego_pose_geom", "gt_boxes"] if self._is_ground_truth_included else ["ego_pose_geom"]
            ),
            data_group_field_array_children=["cams"],
            current_name=None,
        )

        # Ego-pose
        data_ego_pose = data["ego_pose_geom"]
        data_ego_pose.check_has_children(
            data_field_children=["lidar_ego_pose", "lidar_ego_pose_inv"], current_name="['ego_pose_geom']"
        )

        # Cameras
        num_cams = len(data["cams"])
        for c in range(num_cams):

            cam_data = data["cams"][c]
            cam_data.check_has_children(
                data_field_children=["image", "timestamp"],
                data_group_field_children=(
                    ["cam_geometry", "gt_boxes_2d"] if self._is_ground_truth_included else ["cam_geometry"]
                ),
                current_name=f"['cams'][{c}]",
            )
            cam_geom_data = cam_data["cam_geometry"]
            cam_geom_data.check_has_children(
                data_field_children=["extr_lidar2img", "intr_lidar2img", "lidar2img"],
                current_name=f"['cams'][{c}]['cam_geometry']",
            )

            # If needed, also check image ground truth
            if self._is_ground_truth_included:
                img_gt_data = cam_data["gt_boxes_2d"]
                img_gt_data.check_has_children(
                    data_field_children=["bboxes", "centers", "depths", "categories"],
                    current_name=f"['cams'][{c}]['gt_boxes_2d']",
                )

        # If needed, also check 3D ground truth
        if self._is_ground_truth_included:
            gt_data = data["gt_boxes"]
            gt_data.check_has_children(
                data_field_children=[
                    "categories",
                    "sizes",
                    "rotations",
                    "translations",
                    "visibility_level",
                    "orientations",
                    "velocities",
                ],
                current_name="'[gt_boxes]'",
            )
