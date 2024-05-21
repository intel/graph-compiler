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

#ifndef GC_VERSION_H
#define GC_VERSION_H

#if !defined(GC_VERSION_MAJOR) || !defined(GC_VERSION_MINOR) || !defined(GC_VERSION_PATCH)
// define an invalid version if it wasn't defined by CMake
#include <limits>
#define GC_VERSION_MAJOR std::numeric_limits<uint8_t>::max()
#define GC_VERSION_MINOR std::numeric_limits<uint8_t>::max()
#define GC_VERSION_PATCH std::numeric_limits<uint8_t>::max()
#endif
#ifndef GC_VERSION_HASH
#define GC_VERSION_HASH "N/A"
#endif

#endif
