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

#include <memory>
#include <new>
#include <string_view>

#include "JsonParser.h"
#include "gc/ExecutionEngine/Driver/Driver.h"
#include "gc_version.h"

#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

#include "graph/backend/elyzor/include/dnnl_graph_compiler.h"

#if defined _WIN32 || defined __CYGWIN__
#define GC_DLL_EXPORT __declspec(dllexport)
#else
#define GC_DLL_EXPORT __attribute__((visibility("default")))
#endif

// dnnl_graph_compiler.h interface implementation.

using namespace mlir;

extern "C" {
extern int gc_runtime_keep_alive;
}

struct dnnl_graph_compiler_executable {
  dnnl_graph_compiler_executable(MLIRContext &context,
                                 const std::string_view &json)
      : inputIds(), outputIds(), jmod() {
    auto mod = JsonParser::parse(context, json, inputIds, outputIds, strides);
    auto jmod = gc::JitModule::create(mod);

    if (!static_cast<bool>(jmod)) {
      auto err = jmod.takeError();
      llvm::errs() << err;
      llvm::consumeError(std::move(err));
      std::string msg("Failed to create JitModule: ");
      llvm::raw_string_ostream(msg) << err;
      throw std::runtime_error(msg);
    }

    this->jmod = *jmod;
  }

  void execute(dnnl_graph_compiler_tensor *inputs,
               dnnl_graph_compiler_tensor *outputs) const {
    std::vector<MemRefWrapper> memRefs;
    memRefs.reserve(inputIds.size() + outputIds.size());
    for (auto &pair : {std::make_pair(&inputIds, inputs),
                       std::make_pair(&outputIds, outputs)}) {
      auto ids = pair.first;
      auto tensors = pair.second;
      for (size_t i = 0, n = ids->size(); i < n; i++) {
        auto id = (*ids)[i];
        dnnl_graph_compiler_tensor *tensor;

        if (tensors[i].id == id) {
          tensor = &tensors[i];
        } else {
          // The order of inputs/outputs may not match the function args order.
          tensor = nullptr;
          for (size_t j = 0; j < n; j++) {
            if (tensors[j].id == id) {
              tensor = &tensors[j];
              break;
            }
          }
          if (!tensor) {
            throw std::invalid_argument("Tensor not found");
          }
        }

        auto s = strides.find((*ids)[i]);
        memRefs.emplace_back(tensor, s == strides.end() ? nullptr : &s->second);
      }
    }

    llvm::SmallVector<void *> ptrs;
    ptrs.reserve(memRefs.size());
    for (auto &memRef : memRefs) {
      ptrs.push_back(&memRef);
    }
    jmod->call(ptrs.data(), ptrs.size());
  }

private:
  llvm::SmallVector<size_t> inputIds;
  llvm::SmallVector<size_t> outputIds;
  std::unordered_map<std::size_t, Strides> strides;
  std::shared_ptr<gc::JitModule> jmod;

  // C-compatible data wrapper -
  // https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission
  struct MemRefWrapper {
    void *basePtr;
    void *data;
    int64_t offset = 0;
    int64_t dimsAndStrides[2 * DNNL_MAX_NDIMS];

    MemRefWrapper(dnnl_graph_compiler_tensor *tensor, const Strides *strides)
        // We assume, that the data is aligned, thus basePtr == data.
        : basePtr(tensor->data), data(tensor->data) {
      if (tensor->ndims > DNNL_MAX_NDIMS) {
        throw std::invalid_argument("Number of dimensions > DNNL_MAX_NDIMS");
      }

      std::copy(tensor->dims, tensor->dims + tensor->ndims, dimsAndStrides);
      if (strides) {
        std::copy(strides->begin(), strides->end(),
                  dimsAndStrides + tensor->ndims);
      } else {
        std::fill(dimsAndStrides + tensor->ndims,
                  dimsAndStrides + 2 * tensor->ndims, 0);
      }
    }
  };
};

struct dnnl_graph_compiler {
  mutable MLIRContext context;

  explicit dnnl_graph_compiler(llvm::ThreadPoolStrategy &tps)
      : context(gc::initCompilerAndGetDialects(),
                MLIRContext::Threading::DISABLED),
        threadPool(tps) {
    context.setThreadPool(threadPool);
    context.loadAllAvailableDialects();
    // FIXME: keeps GCCPURuntime linked
    gc_runtime_keep_alive = 0;
  }

private:
  llvm::DefaultThreadPool threadPool;
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
    llvm::ThreadPoolStrategy tps;
    tps.ThreadsRequested = (ctx == nullptr) ? 0 : ctx->num_threads;
    *gc = new dnnl_graph_compiler(tps);
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
    *exe = new dnnl_graph_compiler_executable(gc->context,
                                              std::string_view(graph_json));
    return dnnl_success;
  } catch (const std::invalid_argument &e) {
    return dnnl_invalid_graph;
  } catch (const std::logic_error &e) {
    return dnnl_unimplemented;
  } catch (const std::bad_alloc &e) {
    return dnnl_out_of_memory;
  } catch (...) {
    // TODO: Add error handling
    return dnnl_runtime_error;
  }
}

GC_DLL_EXPORT void dnnl_graph_compiler_destroy_executable(
    const struct dnnl_graph_compiler_executable *exe) {
  delete exe;
}

GC_DLL_EXPORT dnnl_status_t dnnl_graph_compiler_execute(
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
