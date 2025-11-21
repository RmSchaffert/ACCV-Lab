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

from accvlab.dali_pipeline_framework.internal_helpers.mini_parser import Lexer, TokenType, Token


def test_lexer():
    input_str = "res = d and not(a_b == -1.5 or -_b2 > 2 and (c < 3 or d > 4)) and c < 3"
    lexer = Lexer(input_str)
    tokens = []
    expected_tokens = [
        Token(TokenType.VARIABLE, "res"),
        Token(TokenType.ASSIGNMENT, "="),
        Token(TokenType.VARIABLE, "d"),
        Token(TokenType.LOGICAL_AND, "and"),
        Token(TokenType.LOGICAL_NOT, "not"),
        Token(TokenType.PARENTHESIS_OPEN, "("),
        Token(TokenType.VARIABLE, "a_b"),
        Token(TokenType.COMPARISON, "=="),
        Token(TokenType.MINUS, "-"),
        Token(TokenType.LITERAL, "1.5"),
        Token(TokenType.LOGICAL_OR, "or"),
        Token(TokenType.MINUS, "-"),
        Token(TokenType.VARIABLE, "_b2"),
        Token(TokenType.COMPARISON, ">"),
        Token(TokenType.LITERAL, "2"),
        Token(TokenType.LOGICAL_AND, "and"),
        Token(TokenType.PARENTHESIS_OPEN, "("),
        Token(TokenType.VARIABLE, "c"),
        Token(TokenType.COMPARISON, "<"),
        Token(TokenType.LITERAL, "3"),
        Token(TokenType.LOGICAL_OR, "or"),
        Token(TokenType.VARIABLE, "d"),
        Token(TokenType.COMPARISON, ">"),
        Token(TokenType.LITERAL, "4"),
        Token(TokenType.PARENTHESIS_CLOSE, ")"),
        Token(TokenType.PARENTHESIS_CLOSE, ")"),
        Token(TokenType.LOGICAL_AND, "and"),
        Token(TokenType.VARIABLE, "c"),
        Token(TokenType.COMPARISON, "<"),
        Token(TokenType.LITERAL, "3"),
        Token(TokenType.EOL, ""),
    ]

    for _ in range(len(expected_tokens)):
        token = lexer.next_token()
        tokens.append(token)

    for token, expected_token in zip(tokens, expected_tokens):
        assert token.type == expected_token.type
        assert token.value == expected_token.value


if __name__ == "__main__":
    pytest.main([__file__])
