#include "dnnl_graph_compiler.hpp"
#include <new>

// dnnl_graph_compiler.h interface implementation.
// TODO: Implement

dnnl_status_t
dnnl_graph_compiler_create(const struct dnnl_graph_compiler_context *ctx,
                           const struct dnnl_graph_compiler **gc) {
  try {
    *gc = new dnnl_graph_compiler(ctx);
    return dnnl_success;
  } catch (const std::bad_alloc &e) {
    return dnnl_out_of_memory;
  } catch (...) {
    // TODO: Add error handling
    return dnnl_runtime_error;
  }
}

void dnnl_graph_compiler_destroy(const struct dnnl_graph_compiler *gc) {
  delete gc;
}

dnnl_status_t
dnnl_graph_compiler_compile(const dnnl_graph_compiler *gc,
                            const char *graph_json,
                            const struct dnnl_graph_compiler_executable **exe) {
  try {
    auto ptr = gc->compile(std::string_view(graph_json));
    *exe = ptr.release();
    return dnnl_success;
  } catch (const std::bad_alloc &e) {
    return dnnl_out_of_memory;
  } catch (...) {
    // TODO: Add error handling
    return dnnl_runtime_error;
  }
}

void dnnl_graph_compiler_destroy_executable(
    const struct dnnl_graph_compiler *gc,
    const struct dnnl_graph_compiler_executable *exe) {
  delete exe;
}

dnnl_status_t
dnnl_graph_compiler_execute(const struct dnnl_graph_compiler *gc,
                            const struct dnnl_graph_compiler_executable *exe,
                            dnnl_graph_compiler_tensor *inputs,
                            dnnl_graph_compiler_tensor *outputs) {
  try {
    exe->execute(inputs, outputs);
    return dnnl_success;
  } catch (const std::bad_alloc &e) {
    return dnnl_out_of_memory;
  } catch (...) {
    // TODO: Add error handling
    return dnnl_runtime_error;
  }
}

std::unique_ptr<const dnnl_graph_compiler_executable>
dnnl_graph_compiler::compile(const std::string_view &graph_json) const {
  // TODO: Implement
  return std::unique_ptr<const dnnl_graph_compiler_executable>(
      new dnnl_graph_compiler_executable());
}

void dnnl_graph_compiler_executable::execute(
    dnnl_graph_compiler_tensor *inputs,
    dnnl_graph_compiler_tensor *outputs) const {
  // TODO: Implement
}
