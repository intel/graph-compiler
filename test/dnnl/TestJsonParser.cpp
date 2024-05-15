#include <gtest/gtest.h>

#include "DnnlTestUtils.h"

#include "graph/backend/elyzor/include/dnnl_graph_compiler.h"

static void compile(const char *fileName) {
  auto json = read_str_resource(fileName);

  const struct dnnl_graph_compiler *gc;
  const struct dnnl_graph_compiler_executable *exe;

  ASSERT_EQ(dnnl_graph_compiler_create(nullptr, &gc), dnnl_success);
  ASSERT_EQ(dnnl_graph_compiler_compile(gc, json.c_str(), &exe), dnnl_success);

  dnnl_graph_compiler_destroy_executable(gc, exe);
  dnnl_graph_compiler_destroy(gc);
}

TEST(TestJsonParser, AddRelu) { compile("add_relu.json"); }
TEST(TestJsonParser, Mpl) { compile("mpl.json"); }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
