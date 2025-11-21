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

from accvlab.dali_pipeline_framework.pipeline import SampleDataGroup
from nvidia.dali import types

# This is a mapping which maps object categories expressed as strings to numeric values. The last entry with key 'None' indicates that any other string will be mapped to
# the respective value (3 in this case). This is similar to the 'default' case in a switch statement.
categories_mapping = {"pedestrian": 0, "bicycle": 1, "vehicle": 2, None: 3}
# Mapping for visibility levels
visibility_mapping = {"level_0": 0, "level_1": 1, "level_2": 2, None: -1}

# This is the root element of our data structure
sample = SampleDataGroup()

# Setup camera data format
cam = SampleDataGroup()
# Camera should contain an image, and the individual elements should be of type float.
cam.add_data_field("image", types.DALIDataType.UINT8, None)
# Apart from the image, each camera will have an image annotation containing multiple fields, including ...
cam_annotation = SampleDataGroup()
# ... bounding boxes for the objects in the image
cam_annotation.add_data_field("bboxes", types.DALIDataType.FLOAT, None)
# ... categories for the objects in the image. Note that we are using the string to numeric mapping defined above. The mapping
# will be automatically applied for the individual elements when assigning data.
cam_annotation.add_data_field("categories", types.DALIDataType.INT32, mapping=categories_mapping)
# ... and a projection matrix
cam_annotation.add_data_field("projection_matrix", types.DALIDataType.FLOAT)
# Add camera annotation to camera
cam.add_data_group_field("cam_annotation", cam_annotation)

# Setup map data format
map_data = SampleDataGroup()
map_data.add_data_field("image", types.DALIDataType.UINT8)
map_data.add_data_field("another_map_example_field", types.DALIDataType.BOOL)

# Setup scene data format, consisting of ...
scene_data = SampleDataGroup()
# ... 3D bounding boxes for the objects in the scene
scene_data.add_data_field("bboxes_3d", types.DALIDataType.FLOAT)
# ... Categories for the objects in the scene (again, with the previously defined mapping)
scene_data.add_data_field("categories", types.DALIDataType.INT32, mapping=categories_mapping)
# ... Visibility levels for the objects (also with a previously defined string to numeric mapping)
scene_data.add_data_field("visibility_levels", types.DALIDataType.INT32, mapping=visibility_mapping)
# ... and Ego-pose of the vehicle
scene_data.add_data_field("ego_pose", types.DALIDataType.FLOAT)
## Add map as part of the scene
scene_data.add_data_group_field("map", map_data)
scene_data.add_data_field("token", types.DALIDataType.STRING)

# Setup lidar data format
lidar_data = SampleDataGroup()
lidar_data.add_data_field("point_cloud", types.DALIDataType.FLOAT)
lidar_data.add_data_field("another_lidar_example_field", types.DALIDataType.INT32)

# Add multiple cameras to the root element
sample.add_data_group_field_array("cam", cam, 2)
# Add scene info to the root element
sample.add_data_group_field("scene", scene_data)
# Add multiple lidars to the root element
sample.add_data_group_field_array("lidar", lidar_data, 1)

print(sample)

# The actual data is set after defining the data format. For example, this will set the camera images and the map image to some (dummy) images
sample["cam"][0]["image"] = np.ones((10, 10)) * 0
sample["cam"][1]["image"] = np.ones((10, 10)) * 1
sample["scene"]["map"]["image"] = np.ones((10, 10)) * 2
sample["scene"]["token"] = "<sample_text_token>"

# If a data field has a mapping , it will automatically be applied. For example, the following code line will store the values [2, 0, 2, 1]
# in sample["cam"][0]["cam_annotation"]["categories"] (according to 'categories_mapping' as defined above)
sample["cam"][0]["cam_annotation"]["categories"] = ["vehicle", "pedestrian", "vehicle", "bicycle"]

# We can print out the data format to see the defined tree as well as which data we have already set
print(sample)

print("print(sampe.field_names_flat): ")
print(sample.field_names_flat)
print("print(sampe.field_types_flat): ")
print(sample.field_types_flat)

image_paths = sample.find_all_occurrences("image")

for path in image_paths:
    res = sample.get_item_in_path(path)
    print(str(path) + ": " + str(res))
    res = res + 10
    res = sample.set_item_in_path(path, res)

for path in image_paths:
    res = sample.get_item_in_path(path)
    print(str(path) + ": " + str(res))

sample.remove_all_occurrences("bboxes")

dictionary = sample.to_dictionary()

print(sample)

print(dictionary)
