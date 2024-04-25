#include <iostream>
#include "test_utils.hpp"
#include "dnnl_graph_compiler.h"

int main() {
    auto json = read_str_resource("mul_quantize.json");
    std::cout << json << std::endl;

    const struct dnnl_graph_compiler_context ctx = {.num_threads = 4};
    const struct dnnl_graph_compiler *gc;
    const struct dnnl_graph_compiler_executable *exe;

    if (dnnl_graph_compiler_create(&ctx, &gc) != dnnl_success) {
        throw std::runtime_error("Failed to create graph compiler!");
    }

    if (dnnl_graph_compiler_compile(gc, json.c_str(), &exe) != dnnl_success) {
        throw std::runtime_error("Failed to compile graph!");
    }

    // Initialize inputs and outputs
    dnnl_graph_compiler_tensor inputs[2];
    dnnl_graph_compiler_tensor outputs[1];
    uint8_t data_buf[160];
    size_t dims[1] = {10};
    inputs[0] = {.id = 0, .ndims = 1, .dims = dims, .data = data_buf};
    inputs[1] = {.id = 1, .ndims = 1, .dims = dims, .data = &data_buf[40]};
    outputs[0] = {.id = 2, .ndims = 1, .dims = dims, .data = &data_buf[80]};

    if (dnnl_graph_compiler_execute(gc, exe, inputs, outputs) != dnnl_success) {
        throw std::runtime_error("Execution failed!");
    }

    dnnl_graph_compiler_destroy_executable(gc, exe);
    dnnl_graph_compiler_destroy(gc);
    return 0;
}
