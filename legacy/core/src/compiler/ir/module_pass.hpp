/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_MODULE_PASS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_MODULE_PASS_HPP

#include "ir_module.hpp"
#include "pass_info_macros.hpp"
#include <memory>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
struct tir_pass_dependency_t;

/**
 * The base abstruct class of all module passes. The pass should not change the
 * input module and the IR in it. However, it is allowed to set the attibutes
 * in the functions of the input module. To modify the module, the pass can
 * copy the module by calling `module->copy()`, modify on the cloned one and
 * return it. The return value of the module_pass_t may/may not be the same
 * memory object of the input, depending on the inplementation
 * */
class SC_INTERNAL_API module_pass_t {
public:
  virtual const char *get_name() const { return nullptr; }
#ifndef NDEBUG
  virtual void get_dependency_info(tir_pass_dependency_t &out) const;
#endif
  virtual const_ir_module_ptr operator()(const_ir_module_ptr f) = 0;
  virtual ~module_pass_t() = default;
};

using module_pass_ptr = std::unique_ptr<module_pass_t>;
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
