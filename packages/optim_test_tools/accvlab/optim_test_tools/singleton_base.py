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


class SingletonBase:
    """Base class for singleton classes.

    It is used to ensure that only a single instance of the class is created.
    It is e.g. used in the following classes:
    - Stopwatch
    - NVTXRangeWrapper
    - DecoratorSwitch
    """

    _instances = {}

    def __new__(cls, *args, **kwargs):
        """Get instance of the class.

        If an instance of the class already exists, return it.
        Otherwise, create a new instance, store it for later use, and return it.
        """
        if not cls in cls._instances:
            obj = super().__new__(cls)
            cls._instances[cls] = obj
        return cls._instances[cls]
