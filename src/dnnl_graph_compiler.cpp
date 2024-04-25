#include "dnnl_graph_compiler.h"

struct dnnl_graph_compiler {
    const struct dnnl_graph_compiler_context ctx;

    explicit dnnl_graph_compiler(const struct dnnl_graph_compiler_context *ctx) : ctx(*ctx) {}
};

struct dnnl_graph_compiler_executable {
};

dnnl_status_t dnnl_graph_compiler_create(
        const struct dnnl_graph_compiler_context *ctx,
        const struct dnnl_graph_compiler **gc) {
    *gc = new dnnl_graph_compiler(ctx);
    return dnnl_success;
}

void dnnl_graph_compiler_destroy(const struct dnnl_graph_compiler *gc) {
    delete gc;
}

dnnl_status_t dnnl_graph_compiler_compile(
        const struct dnnl_graph_compiler *gc, const char *graph_json,
        const struct dnnl_graph_compiler_executable **exe) {
    *exe = new dnnl_graph_compiler_executable();
    return dnnl_success;
}

void dnnl_graph_compiler_destroy_executable(const struct dnnl_graph_compiler *gc,
                                            const struct dnnl_graph_compiler_executable *exe) {
    delete exe;
}

dnnl_status_t dnnl_graph_compiler_execute(
        const struct dnnl_graph_compiler *gc,
        const struct dnnl_graph_compiler_executable *exe,
        dnnl_graph_compiler_tensor *inputs,
        dnnl_graph_compiler_tensor *outputs) {
    return dnnl_success;
}
