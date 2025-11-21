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

import numpy as np
import shapely


class BboxProjector:

    @staticmethod
    def project_3d_boxes_to_2d(
        lidar_to_ego,
        ego_to_world_lidar,
        objects_in_lidar_coords,
        cam_extrinsics,
        cam_intrinsics,
        ego_to_world_cam,
        objects_annotations,
        image_size,
    ):
        '''Project 3D bounding boxes to 2D.

        The bounding boxes need to be in the format as used in the NuScenes dataset.

        Args:
            lidar_to_ego: Transformation from lidar to ego coordinates
            ego_to_world_lidar: Transformation from ego to world coordinates
            objects_in_lidar_coords: List of objects in lidar coordinates
            cam_extrinsics: Camera extrinsics
            cam_intrinsics: Camera intrinsics
            ego_to_world_cam: Transformation from ego to world coordinates
            objects_annotations: List of annotations
            image_size: Image size

        Returns:
            Dictionary with the following keys:
              - bboxes: 2D bounding boxes in the format [x1, y1, x2, y2]
              - centers: 2D centers of the bounding boxes
              - depths: depths of the bounding boxes
              - categories: categories of the bounding boxes
              - num_radar_pts: number of radar points in the bounding box
              - num_lidar_pts: number of lidar points in the bounding box
        '''

        # Get the transformation from lidar to camera coordinates
        correction_lidar_to_cam_timing_in_ego = np.linalg.inv(ego_to_world_cam) @ ego_to_world_lidar
        cam_extrinsics_rel_lidar = cam_extrinsics @ correction_lidar_to_cam_timing_in_ego @ lidar_to_ego

        bboxes = []
        centers = []
        depths = []
        categories = []
        num_radar_points = []
        num_lidar_points = []
        for obj_lidar, anno in zip(objects_in_lidar_coords, objects_annotations):
            cube = BboxProjector._get_points(obj_lidar.center, obj_lidar.wlh, obj_lidar.rotation_matrix)
            points_2d, center_2d, depth = BboxProjector._project(
                cube, obj_lidar.center, cam_extrinsics_rel_lidar, cam_intrinsics
            )
            if points_2d is not None and center_2d is not None:
                rect_2d = BboxProjector._get_rect(points_2d, image_size)
                if rect_2d is not None:
                    bboxes.append(rect_2d)
                    centers.append(center_2d)
                    depths.append(depth)
                    categories.append(obj_lidar.name)
                    num_radar_points.append(anno["num_radar_pts"])
                    num_lidar_points.append(anno["num_lidar_pts"])

        if len(bboxes) > 0:
            projected_annotations = {
                "bboxes": np.stack(bboxes, axis=0),
                "centers": np.stack(centers, axis=0),
                "depths": np.array(depths),
                "categories": categories,
                "num_radar_pts": num_radar_points,
                "num_lidar_pts": num_lidar_points,
            }
        else:
            projected_annotations = {
                "bboxes": np.zeros((0, 4)),
                "centers": np.zeros((0, 2)),
                "depths": np.zeros((0,)),
                "categories": [],
                "num_radar_pts": [],
                "num_lidar_pts": [],
            }
        return projected_annotations

    @staticmethod
    def _get_points(center, size, rotation_matrix):
        size_to_use = np.array([size[1], size[0], size[2]])
        min = -size_to_use * 0.5
        max = size_to_use * 0.5
        points = np.array(
            [
                [min[0], min[1], min[2]],
                [min[0], min[1], max[2]],
                [min[0], max[1], min[2]],
                [min[0], max[1], max[2]],
                [max[0], min[1], min[2]],
                [max[0], min[1], max[2]],
                [max[0], max[1], min[2]],
                [max[0], max[1], max[2]],
            ]
        ).transpose()

        points = rotation_matrix @ points

        points += np.expand_dims(center, axis=1)
        return points

    @staticmethod
    def _get_rect(projected_points, image_size):
        image_rect = shapely.geometry.box(0, 0, image_size[0], image_size[1])
        points = [projected_points[:, i] for i in range(projected_points.shape[1])]
        projection = shapely.geometry.MultiPoint(points).convex_hull
        intersection = image_rect.intersection(projection)
        intersection_coords = np.array([crd for crd in intersection.exterior.coords])

        if np.prod(intersection_coords.shape) > 0:
            rect = np.zeros(4)
            rect[0] = np.min(intersection_coords[:, 0])
            rect[1] = np.min(intersection_coords[:, 1])
            rect[2] = np.max(intersection_coords[:, 0])
            rect[3] = np.max(intersection_coords[:, 1])
        else:
            rect = None
        return rect

    @staticmethod
    def _project(points, center, extrinsics, intrinsics):
        num_points = points.shape[1]
        ones = np.ones((1, num_points), dtype=points.dtype)
        points_homog = np.concatenate((points, ones), axis=0)

        points_cam = extrinsics @ points_homog

        to_keep = points_cam[2, :] > 0.0
        kept_points = points_cam[:3, to_keep]
        num_remaining = kept_points.shape[1]

        if num_remaining > 1:
            projected_homog = intrinsics @ kept_points
            projected = projected_homog[0:2, :] / projected_homog[2, :]
        else:
            projected = None

        center_homog = np.array([center[0], center[1], center[2], 1.0])
        center_cam = extrinsics @ center_homog
        depth = center_cam[2]
        if depth > 0.0:
            center_2d_homog = intrinsics @ center_cam[0:3]
            center_2d = center_2d_homog[0:2] / center_2d_homog[2]
        else:
            center_2d = None

        return projected, center_2d, depth
