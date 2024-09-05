/*
 * Copyright (C) 2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Bindings/Python/PybindAdaptors.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
// x86 or x86_64 specific code
void cpuid(int info[4], int leaf, int subleaf) {
  __asm__ __volatile__("cpuid"
                       : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]),
                         "=d"(info[3])
                       : "a"(leaf), "c"(subleaf));
}

std::vector<int> getCacheSizes() {
  int info[4];
  cpuid(info, 0, 0);
  int nIds = info[0];
  int caches[3] = {};
  for (int i = 0; i <= nIds; ++i) {
    cpuid(info, 4, i);
    int cacheType = info[0] & 0x1F;
    if (cacheType == 0) {
      break;
    }

    int cacheLevel = (info[0] >> 5) & 0x7;
    int cacheLinesPerTag = ((info[1] >> 0) & 0xFFF) + 1;
    int cacheAssociativity = ((info[1] >> 12) & 0x3FF) + 1;
    int cachePartitions = ((info[1] >> 22) & 0x3FF) + 1;
    int cacheSets = info[2] + 1;
    int cacheSize =
        cacheLinesPerTag * cacheAssociativity * cachePartitions * cacheSets;
    if (cacheLevel >= 1 && cacheLevel <= 3) {
      caches[cacheLevel - 1] = cacheSize;
    }
  }
  return std::vector<int>(std::begin(caches), std::end(caches));
}

bool isFeatureSupported(int function_id, int register_idx, int bit) {
  int info[4];
  cpuid(info, function_id, 0);
  return (info[register_idx] & (1 << bit)) != 0;
}

int getMaxVectorWidth() {
  if (isFeatureSupported(7, 1, 16)) { // Check for AVX-512F support
    return 512;
  } else if (isFeatureSupported(1, 2, 28)) { // Check for AVX support
    return 256;
  } else if (isFeatureSupported(1, 3, 25)) { // Check for SSE support
    return 128;
  }
  return 64; // Default to 64 if none of the above features are supported
}
#else
std::vector<int> getCacheSizes() { return {}; }

int getMaxVectorWidth { return 0; }
#endif

PYBIND11_MODULE(_cpuinfo, m) {
  m.doc() = "Graph-compiler MLIR Python binding";
  m.def("get_cache_sizes", &getCacheSizes, "Get CPU L1,L2,L3 cache size");
  m.def("get_max_vector_width", &getMaxVectorWidth,
        "Get CPU supported max vector width");
}