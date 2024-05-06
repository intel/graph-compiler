#include "dnnl_graph_compiler.h"
#include "dnnl_test_utils.hpp"
#include <gtest/gtest.h>

TEST(dnnl_graph_compiler, c_interface) {
  auto json = read_str_resource("mul_quantize.json");

  const struct dnnl_graph_compiler_context ctx = {.num_threads = 4};
  const struct dnnl_graph_compiler *gc;
  const struct dnnl_graph_compiler_executable *exe;

  ASSERT_EQ(dnnl_graph_compiler_create(&ctx, &gc), dnnl_success);

  ASSERT_EQ(dnnl_graph_compiler_compile(gc, json.c_str(), &exe), dnnl_success);

  // Initialize inputs and outputs
  dnnl_graph_compiler_tensor inputs[2];
  dnnl_graph_compiler_tensor outputs[1];
  uint8_t data_buf[160];
  size_t dims[1] = {10};
  inputs[0] = {.id = 0, .ndims = 1, .dims = dims, .data = data_buf};
  inputs[1] = {.id = 1, .ndims = 1, .dims = dims, .data = &data_buf[40]};
  outputs[0] = {.id = 2, .ndims = 1, .dims = dims, .data = &data_buf[80]};

  ASSERT_EQ(dnnl_graph_compiler_execute(gc, exe, inputs, outputs),
            dnnl_success);

  dnnl_graph_compiler_destroy_executable(gc, exe);
  dnnl_graph_compiler_destroy(gc);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
