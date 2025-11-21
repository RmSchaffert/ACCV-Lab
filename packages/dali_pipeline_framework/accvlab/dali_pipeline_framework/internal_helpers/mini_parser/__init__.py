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

"""
This module contains the mini-parser for the DALI pipeline. See :class:`Parser` for more details on the parser
and :class:`AST` and its subclasses for more details on the abstract syntax tree which is used as the
output of the parser.

The parser is used internally in the :class:`~accvlab.dali_pipeline_framework.processing_steps.AnnotationConditionEval`
processing step.
"""

from .ast import AST, Assignment, Literal, Variable, Comparison, Or, And, Not, UnaryMinus
from .parser import Parser
from .lexer import Lexer, TokenType, Token

__all__ = [
    "Parser",
    "AST",
    "Assignment",
    "Literal",
    "Variable",
    "Comparison",
    "Or",
    "And",
    "Not",
    "UnaryMinus",
    "Lexer",
    "TokenType",
    "Token",
]
