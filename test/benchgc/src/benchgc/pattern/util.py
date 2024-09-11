################################################################################
# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0
################################################################################

from typing import List


def to_int_list(s: str) -> List[int]:
    """
    Parsing the cmd for list of int values

    Args:
        s (str): int values in cmd, example: 2x3x4

    Returns:
        List[int]: int values in list, example: [2, 3, 4]
    """
    if not s or len(s) == 0:
        return []
    return [int(i) for i in s.strip().split("x")]


def to_bool_list(s: str) -> List[bool]:
    """
    Parsing the cmd for list of bool values

    Args:
        s (str): bools in cmd, example: 1x0x1

    Returns:
        List[bool]: bools in list, example: [True, False, True]
    """
    if not s or len(s) == 0:
        return []
    return [bool(int(i)) for i in s.strip().split("x")]
