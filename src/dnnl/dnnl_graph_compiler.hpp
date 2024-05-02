#ifndef GRAPH_COMPILER_HPP
#define GRAPH_COMPILER_HPP

#include "dnnl_graph_compiler.h"
#include <memory>
#include <string_view>

struct dnnl_graph_compiler_executable {
  // TODO: Implement

  void execute(dnnl_graph_compiler_tensor *inputs,
               dnnl_graph_compiler_tensor *outputs) const;
};

struct dnnl_graph_compiler {
  const dnnl_graph_compiler_context ctx;

  explicit dnnl_graph_compiler(const dnnl_graph_compiler_context *context)
      // TODO: Initialize ctx with context or defaults if context is nullptr
      : ctx() {}

  std::unique_ptr<const dnnl_graph_compiler_executable>
  compile(const std::string_view &graph_json) const;
};

#endif // GRAPH_COMPILER_HPP
