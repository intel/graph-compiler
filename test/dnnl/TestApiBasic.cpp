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

#include <gtest/gtest.h>

#include "DnnlTestUtils.h"
#include "gc_version.h"
#include "graph/backend/elyzor/include/dnnl_graph_compiler.h"

TEST(TestApiBasic, basicWorkflow) {
  auto json = readStrResource("add.json");

  const struct dnnl_graph_compiler_context ctx = {.num_threads = 4};
  const struct dnnl_graph_compiler *gc;
  const struct dnnl_graph_compiler_executable *exe;

  ASSERT_EQ(dnnl_graph_compiler_create(&ctx, &gc), dnnl_success);

  ASSERT_EQ(dnnl_graph_compiler_compile(gc, json.c_str(), &exe), dnnl_success);

  // Initialize inputs and outputs
  dnnl_graph_compiler_tensor inputs[2];
  dnnl_graph_compiler_tensor outputs[1];
  float arg1[128]{1.f};
  float arg2[128];
  float arg3[128];
  int64_t dims[1] = {128};
  inputs[0] = {.id = 0, .ndims = 1, .dims = dims, .data = arg1};
  inputs[1] = {.id = 1, .ndims = 1, .dims = dims, .data = arg2};
  outputs[0] = {.id = 2, .ndims = 1, .dims = dims, .data = arg3};
  for (size_t i = 0; i < 128; i++) {
    arg2[i] = i;
  }

  ASSERT_EQ(dnnl_graph_compiler_execute(exe, inputs, outputs), dnnl_success);

  dnnl_graph_compiler_destroy_executable(exe);
  dnnl_graph_compiler_destroy(gc);

  for (size_t i = 0; i < 128; i++) {
    ASSERT_FLOAT_EQ(arg3[i], arg1[i] + arg2[i]);
  }
}

TEST(TestApiBasic, get_version) {
  auto v = dnnl_graph_compiler_get_version();

  ASSERT_NE(v, nullptr);

  ASSERT_EQ(v->api_version.major, DNNL_GC_API_V_MAJOR);
  ASSERT_EQ(v->api_version.minor, DNNL_GC_API_V_MINOR);
  ASSERT_EQ(v->api_version.patch, DNNL_GC_API_V_PATCH);
  ASSERT_STREQ(v->api_version.hash, DNNL_GC_API_V_HASH);

  // check if the version is valid
  ASSERT_NE(v->gc_version.major, std::numeric_limits<uint8_t>::max());
  ASSERT_NE(v->gc_version.minor, std::numeric_limits<uint8_t>::max());
  ASSERT_NE(v->gc_version.patch, std::numeric_limits<uint8_t>::max());

  ASSERT_EQ(v->gc_version.major, GC_VERSION_MAJOR);
  ASSERT_EQ(v->gc_version.minor, GC_VERSION_MINOR);
  ASSERT_EQ(v->gc_version.patch, GC_VERSION_PATCH);
  ASSERT_STREQ(v->gc_version.hash, GC_VERSION_HASH);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
