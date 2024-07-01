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

#include <functional>
#include <memory>
#include <new>
#include <sstream>
#include <string_view>
#include <unordered_map>

#if __cplusplus > 202002L
#include <stdfloat>
#else
namespace std {
#if defined(__SIZEOF_FLOAT__) && __SIZEOF_FLOAT__ == 4
using float32_t = float;
#elif defined(__SIZEOF_DOUBLE__) && __SIZEOF_DOUBLE__ == 4
using float32_t = double;
#else
static_assert(false, "Unable to determine 32-bit floating point type");
#endif
} // namespace std
#endif

#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphOps.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "dnnl_types.h"
#include "graph/utils/json.hpp"

using Strides = llvm::SmallVector<int64_t, DNNL_MAX_NDIMS>;

class JsonParser {
  dnnl::impl::graph::utils::json::json_reader_t _reader;
  mlir::OpBuilder _builder;
  mlir::Location _loc;
  mlir::Block *_entryBlock;
  llvm::SmallVector<size_t> &_inputIds;
  std::unordered_map<std::size_t, Strides> _strides;
  // Function input and operations output values. Used to connect the
  // operations inputs and outputs.
  std::unordered_map<std::size_t, mlir::Value> _valueMap;
  // Temporary value holders, used by the parser
  llvm::SmallVector<mlir::Value> _operands;
  llvm::SmallVector<mlir::Type> _resultTypes;
  llvm::SmallVector<mlir::NamedAttribute> _attributes;
  std::string _str;
  std::string _str2;
  std::size_t _uS;
  std::int64_t _i64;
  std::float32_t _f32;
  std::vector<size_t> _uaS;
  std::vector<std::int64_t> _ia64;
  std::vector<std::int64_t> _ia642;
  std::vector<std::float32_t> _fa32;

  JsonParser(mlir::MLIRContext &context, std::istream &stream,
             llvm::SmallVector<size_t> &inputIds)
      : _reader(&stream), _builder(&context), _loc(_builder.getUnknownLoc()),
        _inputIds(inputIds), _strides(), _valueMap(), _operands(),
        _resultTypes(), _attributes(), _str(), _str2(), _uS(), _i64(), _f32(),
        _uaS(), _ia64(), _ia642(), _fa32() {
    // Creating a dummy function since we don't know the actual type yet.
    auto func = _builder.create<mlir::func::FuncOp>(
        _loc, "tmp", _builder.getFunctionType({}, {}));
    _entryBlock = func.addEntryBlock();
    _builder.setInsertionPointToStart(_entryBlock);
  }

  mlir::ModuleOp parse(llvm::SmallVector<size_t> &outputIds,
                       std::unordered_map<std::size_t, Strides> &strides);
  void readOp();
  mlir::Attribute readAttr();
  mlir::Type readTensorType();

  template <typename Err>
  void throwErr(const char *msgSfx, const char *msgPref = nullptr,
                std::string *str = nullptr) {
    if (str == nullptr) {
      str = &_str;
    }
    str->insert(0, msgSfx);
    if (msgPref != nullptr) {
      str->insert(0, msgSfx);
    }
#ifndef NDEBUG
    std::cerr << "[JsonParser] " << str->c_str() << std::endl;
#endif
    throw Err(*str);
  }

  void throwUnrecognizedKey(std::string *str = nullptr) {
    throwErr<std::invalid_argument>("Unrecognized key: ", nullptr, str);
  }

  inline void readKey(const char *key, std::string *str = nullptr) {
    if (str == nullptr) {
      str = &_str;
    }
    if (!_reader.next_object_item(str)) {
      *str = key;
      throwErr<std::invalid_argument>("Key expected: ", nullptr, str);
    }
    if (*str != key) {
      throwUnrecognizedKey(str);
    }
  }

  template <typename T, template <typename...> class Container, typename... Any>
  inline void readNumArray(Container<T, Any...> &c) {
    _reader.begin_array();
    for (T value; _reader.next_array_item();) {
      _reader.read_number(&value);
      c.push_back(value);
    }
  }

  // Operation builders map
  using OpBuilderFn = mlir::ResultRange (*)(JsonParser &);
#define GC_OP(name, type)                                                      \
  {                                                                            \
    name, [](JsonParser &p) -> mlir::ResultRange {                             \
      return p._builder                                                        \
          .create<type>(p._loc, p._resultTypes, p._operands, p._attributes)    \
          ->getResults();                                                      \
    }                                                                          \
  }
  std::unordered_map<std::string, OpBuilderFn> _opBuilders{
      GC_OP("Add", mlir::onednn_graph::AddOp),
      GC_OP("Divide", mlir::onednn_graph::DivOp),
      GC_OP("MatMul", mlir::onednn_graph::MatMulOp),
      GC_OP("Multiply", mlir::onednn_graph::MulOp),
      GC_OP("Pow", mlir::onednn_graph::PowOp),
      GC_OP("ReduceMean", mlir::onednn_graph::ReduceMeanOp),
      GC_OP("ReduceSum", mlir::onednn_graph::ReduceSumOp),
      GC_OP("ReLU", mlir::onednn_graph::ReLUOp),
      GC_OP("Sigmoid", mlir::onednn_graph::SigmoidOp),
      GC_OP("Subtract", mlir::onednn_graph::SubOp),
      GC_OP("Typecast", mlir::onednn_graph::TypeCastOp),
  };
#undef GC_OP

  // Data types map
  using GetTypeFn = mlir::Type (*)(mlir::OpBuilder &);
#define GC_DTYPE(name, expr)                                                   \
  {                                                                            \
    name, [](mlir::OpBuilder &b) -> mlir::Type { return expr; }                \
  }
  std::unordered_map<std::string, GetTypeFn> _dtypes{
      GC_DTYPE("f16", b.getF16Type()),
      GC_DTYPE("bf16", b.getBF16Type()),
      GC_DTYPE("f32", b.getF32Type()),
      GC_DTYPE("s32", b.getI32Type()),
      GC_DTYPE("s8", b.getIntegerType(8, false)),
      GC_DTYPE("u8", b.getIntegerType(8, true)),
      GC_DTYPE("f64", b.getF64Type()),
      GC_DTYPE("boolean", b.getI1Type()),
      GC_DTYPE("f8_e5m2", b.getFloat8E5M2Type()),
      GC_DTYPE("f8_e4m3", b.getFloat8E4M3FNType()),
      GC_DTYPE("s4", b.getIntegerType(4, false)),
      GC_DTYPE("u4", b.getIntegerType(4, true)),
  };
#undef GC_DTYPE

public:
  /**
   * @brief Parse the oneDNN JSON and convert to MLIR module.
   *
   * @param context MLIR context.
   * @param json JSON string containing the oneDNN graph.
   * @param inputIds Input tensor IDs are added to this vector.
   * @param outputIds Output tensor IDs are added to this vector.
   * @param strides Strides for each tensor are added to this map.
   * @return The resulting MLIR module.
   */
  static mlir::ModuleOp
  parse(mlir::MLIRContext &context, const std::string_view &json,
        llvm::SmallVector<size_t> &inputIds,
        llvm::SmallVector<size_t> &outputIds,
        std::unordered_map<std::size_t, Strides> &strides) {
    std::istringstream stream(json.data());
    JsonParser parser(context, stream, inputIds);
    return parser.parse(outputIds, strides);
  }
};
