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

from accvlab.dali_pipeline_framework.internal_helpers.mini_parser import Parser


def test_parser():
    input_str = "res = d and not(a_b == -1.5 or -_b2 > 2 and (c < 3 or d > 4)) and c < 3"
    parser = Parser(input_str)
    ast_tree = parser.parse()

    expected_ast_tree = "'res' = <{{'d' and not [{('a_b' == -(1.5)) or {(-('_b2') > 2) and {('c' < 3) or ('d' > 4)}}}]} and ('c' < 3)}>"

    assert str(ast_tree) == expected_ast_tree


if __name__ == "__main__":
    pytest.main([__file__])
