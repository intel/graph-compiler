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

import math
from abc import ABC, abstractmethod
from typing import List


class ConfigFilter(ABC):
    """
    A class used to help filter out unseen configs.
    """

    @abstractmethod
    def already_met(self, v: List[int]) -> bool:
        """Check if the config has been met before."""

    @abstractmethod
    def add(self, v: List[int]):
        """Add the config to the filter."""

    @abstractmethod
    def save(self):
        """Save the satus of the filter."""

    @abstractmethod
    def load(self, data):
        """Load the status of the filter."""


class BloomFilter(ConfigFilter):
    """Bloom Filter"""

    def __init__(self, num_samples: int, err_rate: float):
        self.num_bits = int(-(num_samples * math.log(err_rate)) / (math.log(2) ** 2))
        self.num_hashes = int((self.num_bits / num_samples) * math.log(2))
        self.bit_array = [0] * self.num_bits

    def already_met(self, v):
        for i in range(int(self.num_hashes)):
            try:
                import mmh3
            except ImportError:
                raise ImportError("Please install mmh3 package")
            hash_v = mmh3.hash(v, i) % self.num_bits
            if self.bit_array[hash_v] == 0:
                return False
        return True

    def add(self, v):
        for i in range(int(self.num_hashes)):
            try:
                import mmh3
            except ImportError:
                raise ImportError("Please install mmh3 package")
            hash_v = mmh3.hash(v, i) % self.num_bits
            self.bit_array[hash_v] = 1

    def save(self):
        return self.bit_array

    def load(self, data):
        self.bit_array = data


class HashSetFilter(ConfigFilter):
    """Fliter based on HashSet"""

    def __init__(self):
        self.data = set()

    def add(self, v):
        self.data.add(tuple(v))

    def already_met(self, v: List[int]) -> bool:
        return tuple(v) in self.data

    def save(self):
        return self.data

    def load(self, data):
        self.data.clear()
        for item in data:
            self.add(item)