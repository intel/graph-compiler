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

#include "dnnl_graph_compiler.h"
#include "gc_version.h"
#include <memory>
#include <new>
#include <string_view>

#if defined _WIN32 || defined __CYGWIN__
#define GC_DLL_EXPORT __declspec(dllexport)
#else
#define GC_DLL_EXPORT __attribute__((visibility("default")))
#endif

// dnnl_graph_compiler.h interface implementation.
// TODO: Implement.

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

  [[nodiscard]] std::unique_ptr<const dnnl_graph_compiler_executable>
  compile(const std::string_view &graph_json) const;
};

GC_DLL_EXPORT const dnnl_graph_compiler_version *
dnnl_graph_compiler_get_version(void) {
  static const dnnl_graph_compiler_version ver = {
      .api_version = {DNNL_GC_API_V_MAJOR, DNNL_GC_API_V_MINOR,
                      DNNL_GC_API_V_PATCH,
                      DNNL_GC_API_V_HASH}, // version defined by oneDNN
      .gc_version = {
          GC_VERSION_MAJOR, GC_VERSION_MINOR, GC_VERSION_PATCH,
          GC_VERSION_HASH}}; // version defined by graph compiler itself
  return &ver;
}

GC_DLL_EXPORT dnnl_status_t
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

GC_DLL_EXPORT void
dnnl_graph_compiler_destroy(const struct dnnl_graph_compiler *gc) {
  delete gc;
}

GC_DLL_EXPORT dnnl_status_t dnnl_graph_compiler_compile(
    const dnnl_graph_compiler *gc, const char *graph_json,
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

GC_DLL_EXPORT void dnnl_graph_compiler_destroy_executable(
    const struct dnnl_graph_compiler *gc,
    const struct dnnl_graph_compiler_executable *exe) {
  delete exe;
}

GC_DLL_EXPORT dnnl_status_t dnnl_graph_compiler_execute(
    const struct dnnl_graph_compiler *gc,
    const struct dnnl_graph_compiler_executable *exe,
    dnnl_graph_compiler_tensor *inputs, dnnl_graph_compiler_tensor *outputs) {
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
