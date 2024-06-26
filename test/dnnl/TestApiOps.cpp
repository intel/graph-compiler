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

#include <cmath>
#include <gtest/gtest.h>

#include <llvm/ADT/APFloat.h>

#include "DnnlTestUtils.h"
#include "graph/backend/elyzor/include/dnnl_graph_compiler.h"

static void exec(const char *fileName, dnnl_graph_compiler_tensor *inputs,
                 dnnl_graph_compiler_tensor *outputs) {
  auto json = readStrResource(fileName);
  const struct dnnl_graph_compiler *gc;
  const struct dnnl_graph_compiler_executable *exe;
  ASSERT_EQ(dnnl_graph_compiler_create(nullptr, &gc), dnnl_success);
  ASSERT_EQ(dnnl_graph_compiler_compile(gc, json.c_str(), &exe), dnnl_success);
  ASSERT_EQ(dnnl_graph_compiler_execute(exe, inputs, outputs), dnnl_success);
  dnnl_graph_compiler_destroy_executable(exe);
  dnnl_graph_compiler_destroy(gc);
}

TEST(TestApiOps, div) {
  dnnl_graph_compiler_tensor inputs[2];
  dnnl_graph_compiler_tensor outputs[1];
  int64_t dims[3] = {64, 32, 32};
  std::float32_t arg1[64][32][32];
  std::float32_t arg2[64][32][32];
  std::float32_t arg3[64][32][32];
  inputs[0] = {.id = 0, .ndims = 3, .dims = dims, .data = arg1};
  inputs[1] = {.id = 1, .ndims = 3, .dims = dims, .data = arg2};
  outputs[0] = {.id = 2, .ndims = 3, .dims = dims, .data = arg3};
  for (auto i = 0; i < 64; i++) {
    for (auto j = 0; j < 32; j++) {
      for (auto k = 0; k < 32; k++) {
        arg1[i][j][k] = static_cast<std::float32_t>(i * 32 * 32 + j * 32 + k);
        arg2[i][j][k] = arg1[i][j][k] + 1;
      }
    }
  }

  exec("div.json", inputs, outputs);

  for (auto i = 0; i < 64; i++) {
    for (auto j = 0; j < 32; j++) {
      for (auto k = 0; k < 32; k++) {
        ASSERT_EQ(arg1[i][j][k] / arg2[i][j][k], arg3[i][j][k]);
      }
    }
  }
}

TEST(TestApiOps, matMul) {
  dnnl_graph_compiler_tensor inputs[3];
  dnnl_graph_compiler_tensor outputs[1];
  int64_t dimsA[2] = {128, 512};
  int64_t dimsB[2] = {128, 64};
  int64_t dimsBias[1] = {64};
  int64_t dimsOut[2] = {512, 64};
  std::float32_t argA[128][512];
  std::float32_t argB[128][64];
  std::float32_t argBias[64];
  std::float32_t argOut[512][64];
  inputs[0] = {.id = 0, .ndims = 2, .dims = dimsA, .data = argA};
  inputs[1] = {.id = 1, .ndims = 2, .dims = dimsB, .data = argB};
  inputs[2] = {.id = 2, .ndims = 1, .dims = dimsBias, .data = argBias};
  outputs[0] = {.id = 3, .ndims = 2, .dims = dimsOut, .data = argOut};

  // Initialize input tensors
  for (auto i = 0; i < 128; i++) {
    for (auto j = 0; j < 512; j++) {
      argA[i][j] = static_cast<std::float32_t>(i + j);
    }
  }
  for (auto i = 0; i < 128; i++) {
    for (auto j = 0; j < 64; j++) {
      argB[i][j] = static_cast<std::float32_t>(i - j);
    }
  }
  for (auto i = 0; i < 64; i++) {
    argBias[i] = static_cast<std::float32_t>(i);
  }

  exec("matmul.json", inputs, outputs);

  // Calculate expected output
  std::float32_t expected[512][64];
  for (auto i = 0; i < 512; i++) {
    for (auto j = 0; j < 64; j++) {
      expected[i][j] = argBias[j];
      for (auto k = 0; k < 128; k++) {
        expected[i][j] += argA[k][i] * argB[k][j]; // transpose_a = true
      }
    }
  }

  // Compare the results
  for (auto i = 0; i < 512; i++) {
    for (auto j = 0; j < 64; j++) {
      ASSERT_EQ(expected[i][j], argOut[i][j]);
    }
  }
}

TEST(TestApiOps, mul) {
  dnnl_graph_compiler_tensor inputs[2];
  dnnl_graph_compiler_tensor outputs[1];
  int64_t dims[1] = {128};
  std::float32_t arg1[128];
  std::float32_t arg2[128];
  std::float32_t arg3[128];
  inputs[0] = {.id = 0, .ndims = 1, .dims = dims, .data = arg1};
  inputs[1] = {.id = 1, .ndims = 1, .dims = dims, .data = arg2};
  outputs[0] = {.id = 2, .ndims = 1, .dims = dims, .data = arg3};
  for (auto i = 0; i < 128; i++) {
    arg1[i] = arg2[i] = static_cast<std::float32_t>(i);
  }

  exec("mul.json", inputs, outputs);

  for (auto i = 0; i < 128; i++) {
    ASSERT_EQ(arg3[i], static_cast<std::float32_t>(i * i));
  }
}

TEST(TestApiOps, sub) {
  dnnl_graph_compiler_tensor inputs[2];
  dnnl_graph_compiler_tensor outputs[1];
  int64_t dims[1] = {128};
  std::float32_t arg1[128];
  std::float32_t arg2[128];
  std::float32_t arg3[128];
  inputs[0] = {.id = 0, .ndims = 1, .dims = dims, .data = arg1};
  inputs[1] = {.id = 1, .ndims = 1, .dims = dims, .data = arg2};
  outputs[0] = {.id = 2, .ndims = 1, .dims = dims, .data = arg3};
  for (auto i = 0; i < 128; i++) {
    arg1[i] = static_cast<std::float32_t>(i);
    arg2[i] = arg1[i] * arg1[i];
  }

  exec("sub.json", inputs, outputs);

  for (auto i = 0; i < 128; i++) {
    ASSERT_EQ(arg3[i], arg1[i] - arg2[i]);
  }
}

TEST(TestApiOps, pow) {
  dnnl_graph_compiler_tensor inputs[1];
  dnnl_graph_compiler_tensor outputs[1];
  int64_t dims[1] = {64};
  std::float32_t arg1[64];
  std::float32_t arg2[64];
  inputs[0] = {.id = 0, .ndims = 1, .dims = dims, .data = arg1};
  outputs[0] = {.id = 1, .ndims = 1, .dims = dims, .data = arg2};
  for (auto i = 0; i < 64; i++) {
    arg1[i] = static_cast<std::float32_t>(i);
  }

  exec("pow.json", inputs, outputs);

  for (auto i = 0; i < 64; i++) {
    ASSERT_EQ(arg1[i] * arg1[i], arg2[i]);
  }
}

TEST(TestApiOps, relu) {
  dnnl_graph_compiler_tensor inputs[1];
  dnnl_graph_compiler_tensor outputs[1];
  int64_t dims[1] = {128};
  std::float32_t arg1[128];
  std::float32_t arg2[128];
  inputs[0] = {.id = 0, .ndims = 1, .dims = dims, .data = arg1};
  outputs[0] = {.id = 1, .ndims = 1, .dims = dims, .data = arg2};

  for (auto i = 0; i < 128; i++) {
    arg1[i] = static_cast<std::float32_t>(i - 64);
  }

  exec("relu.json", inputs, outputs);

  for (auto i = 0; i < 128; i++) {
    ASSERT_EQ(arg1[i] < 0 ? 0 : arg1[i], arg2[i]);
  }
}

TEST(TestApiOps, reduceMean) {
  dnnl_graph_compiler_tensor inputs[1];
  dnnl_graph_compiler_tensor outputs[1];
  int64_t dims1[3] = {16, 64, 32};
  int64_t dims2[3] = {16, 1, 32};
  std::float32_t arg1[16][64][32];
  std::float32_t arg2[16][1][32];
  inputs[0] = {.id = 0, .ndims = 3, .dims = dims1, .data = arg1};
  outputs[0] = {.id = 1, .ndims = 3, .dims = dims2, .data = arg2};

  for (auto i = 0; i < 16; i++) {
    for (auto y = 0; y < 64; y++) {
      for (auto z = 0; z < 32; z++) {
        arg1[i][y][z] = static_cast<std::float32_t>(i * 64 * 32 + y * 32 + z);
      }
    }
  }

  exec("reduce_mean.json", inputs, outputs);

  std::float32_t expected[16][1][32];
  for (auto x = 0; x < 16; x++) {
    for (auto z = 0; z < 32; z++) {
      expected[x][0][z] = 0;
      for (auto y = 0; y < 64; y++) {
        expected[x][0][z] += arg1[x][y][z];
      }
      expected[x][0][z] /= 64;
    }
  }

  for (auto x = 0; x < 16; x++) {
    for (auto z = 0; z < 32; z++) {
      ASSERT_EQ(expected[x][0][z], arg2[x][0][z]);
    }
  }
}

TEST(TestApiOps, reduceSum) {
  dnnl_graph_compiler_tensor inputs[1];
  dnnl_graph_compiler_tensor outputs[1];
  int64_t dims1[3] = {16, 64, 32};
  int64_t dims2[3] = {16, 1, 32};
  std::float32_t arg1[16][64][32];
  std::float32_t arg2[16][1][32];
  inputs[0] = {.id = 0, .ndims = 3, .dims = dims1, .data = arg1};
  outputs[0] = {.id = 1, .ndims = 3, .dims = dims2, .data = arg2};

  for (auto i = 0; i < 16; i++) {
    for (auto y = 0; y < 64; y++) {
      for (auto z = 0; z < 32; z++) {
        arg1[i][y][z] = static_cast<std::float32_t>(i * 64 * 32 + y * 32 + z);
      }
    }
  }

  exec("reduce_sum.json", inputs, outputs);

  std::float32_t expected[16][1][32];
  for (auto x = 0; x < 16; x++) {
    for (auto z = 0; z < 32; z++) {
      expected[x][0][z] = 0;
      for (auto y = 0; y < 64; y++) {
        expected[x][0][z] += arg1[x][y][z];
      }
    }
  }

  for (auto x = 0; x < 16; x++) {
    for (auto z = 0; z < 32; z++) {
      ASSERT_EQ(expected[x][0][z], arg2[x][0][z]);
    }
  }
}

TEST(TestApiOps, sigmoid) {
  dnnl_graph_compiler_tensor inputs[1];
  dnnl_graph_compiler_tensor outputs[1];
  int64_t dims[1] = {128};
  std::float32_t arg1[128];
  std::float32_t arg2[128];
  inputs[0] = {.id = 0, .ndims = 1, .dims = dims, .data = arg1};
  outputs[0] = {.id = 1, .ndims = 1, .dims = dims, .data = arg2};
  for (auto i = 0; i < 128; i++) {
    arg1[i] = static_cast<std::float32_t>(i - 64);
  }

  exec("sigmoid.json", inputs, outputs);

  for (auto i = 0; i < 128; i++) {
    ASSERT_EQ(1.f / (1.f + std::exp(-arg1[i])), arg2[i]);
  }
}

TEST(TestApiOps, typecast) {
  dnnl_graph_compiler_tensor inputs[1];
  dnnl_graph_compiler_tensor outputs[1];
  int64_t dims[1] = {128};
  std::float32_t arg1[128];
  uint16_t arg2[128];
  inputs[0] = {.id = 0, .ndims = 1, .dims = dims, .data = arg1};
  outputs[0] = {.id = 1, .ndims = 1, .dims = dims, .data = arg2};
  for (auto i = 0; i < 128; i++) {
    auto x = i - 64;
    arg1[i] = static_cast<std::float32_t>(x / (std::exp(-x)));
  }

  exec("typecast.json", inputs, outputs);

  for (auto i = 0; i < 128; i++) {
    llvm::APFloat f(arg1[i]);
    bool losesInfo;
    f.convert(llvm::APFloat::IEEEhalf(), llvm::APFloat::rmNearestTiesToEven,
              &losesInfo);
    ASSERT_EQ(static_cast<uint16_t>(f.bitcastToAPInt().getZExtValue()),
              arg2[i]);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
