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

#include <limits>
#include <memory>
#include <sstream>
#include <string_view>

#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"

#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"

#include "JsonParser.h"

mlir::ModuleOp JsonParser::parse() {
  std::vector<size_t> inputPorts;
  bool hasInputPorts = false;
  bool hasOutputPorts = false;
  _reader.begin_object();

  while (_reader.next_object_item(&_str)) {
    if (_str == "version") {
      _reader.read_string(&_str);
      // TODO: Check if the version supported
    } else if (_str == "engine_kind") {
      _reader.read_string(&_str);
      if (_str != "cpu") {
        throwErr<std::logic_error>("Unsupported engine: ");
      }
    } else if (_str == "fpmath_mode") {
      _reader.read_string(&_str);
      if ((_str != "strict") && (_str != "any")) {
        throwErr<std::logic_error>(
            "Unsupported fpmath_mode: ",
            ". Only 'strict' and 'any' are currently supported.");
      }
    } else if (_str == "input_ports") {
      hasInputPorts = true;
      readNumArray(inputPorts);
    } else if (_str == "output_ports") {
      hasOutputPorts = true;
      readNumArray(_outputIds);
    } else if (_str == "graph") {
      _reader.begin_array();
      while (_reader.next_array_item()) {
        readOp();
      }
    } else {
      throwUnrecognizedKey();
    }
  }

  // Check if the input_ports match the expected inputs
  if (hasInputPorts) {
    if (inputPorts.size() != _inputIds.size()) {
      _str = std::to_string(_inputIds.size());
      throwErr<std::invalid_argument>(
          "Length mismatch between input_ports and inputs: ");
    }
    for (auto id : _inputIds) {
      // The order of the inputs could be different
      if (std::find(inputPorts.begin(), inputPorts.end(), id) ==
          inputPorts.end()) {
        _str = std::to_string(id);
        throwErr<std::invalid_argument>("Input not found in input_ports: ");
      }
    }
  }

  if (!hasOutputPorts) {
    // If output_ports is not specified, using the last operation's outputs.
    _outputIds = _uaS;
  }

  // The function return values.
  std::vector<mlir::Value> outputs;
  outputs.reserve(_outputIds.size());
  for (auto id : _outputIds) {
    auto entry = _valueMap.find(id);
    if (entry == _valueMap.end()) {
      _str = std::to_string(id);
      throwErr<std::invalid_argument>("Output value not found: ");
    }
    outputs.push_back(entry->second);
  }
  auto ret = _builder.create<mlir::func::ReturnOp>(_loc, outputs);

  // Creating the final function and moving the entry block.
  mlir::OpBuilder builder(_builder.getContext());
  auto module = builder.create<mlir::ModuleOp>(_loc);
  auto func = builder.create<mlir::func::FuncOp>(
      _loc, "main",
      builder.getFunctionType(_entryBlock->getArgumentTypes(),
                              ret->getOperandTypes()));
  auto entry = func.addEntryBlock();
  _entryBlock->moveBefore(entry);
  entry->erase();
  module.push_back(func);
  return module;
}

void JsonParser::readOp() {
  OpBuilderFn builderFn = nullptr;

  _uaS.clear();
  _operands.clear();
  _attributes.clear();
  _resultTypes.clear();
  _reader.begin_object();

  while (_reader.next_object_item(&_str)) {
    if (_str == "id") {
      // ignore
      _reader.read_number(&_uS);
    } else if (_str == "name") {
      // ignore
      _reader.read_string(&_str);
    } else if (_str == "kind") {
      _reader.read_string(&_str);
      auto fn = _opBuilders.find(_str);
      if (fn == _opBuilders.end()) {
        throwErr<std::logic_error>("Unsupported operation: ");
      }
      builderFn = fn->second;
    } else if (_str == "attrs") {
      _reader.begin_object();
      while (_reader.next_object_item(&_str)) {
        auto name = mlir::StringAttr::get(_builder.getContext(), _str);
        _attributes.emplace_back(name, readAttr());
      }
    } else if (_str == "inputs") {
      _reader.begin_array();
      while (_reader.next_array_item()) {
        auto type = readTensorType();
        auto entry = _valueMap.find(_uS);
        if (entry == _valueMap.end()) {
          // If not found, then this is a function argument.
          auto value = _entryBlock->addArgument(type, _loc);
          _valueMap[_uS] = value;
          _operands.push_back(value);
          _inputIds.push_back(_uS);
        } else {
          if (entry->second.getType() != type) {
            _str = std::to_string(_uS);
            throwErr<std::invalid_argument>("Type mismatch for input: ");
          }
          _operands.push_back(entry->second);
        }
      }
    } else if (_str == "outputs") {
      _reader.begin_array();
      while (_reader.next_array_item()) {
        _resultTypes.push_back(readTensorType());
        _uaS.push_back(_uS);
      }
    } else {
      throwUnrecognizedKey();
    }
  }

  if (builderFn == nullptr) {
    throwErr<std::invalid_argument>("Operation kind is not specified");
  }

  auto outputs = builderFn(*this);
  assert(outputs.size() == _uaS.size());
  auto id = _uaS.begin();
  auto value = outputs.begin();

  for (; id != _uaS.end(); ++id, ++value) {
    if (!_valueMap.emplace(*id, *value).second) {
      _str = std::to_string(*id);
      throwErr<std::invalid_argument>("Duplicate output id: ");
    }
  }
}

inline mlir::Attribute JsonParser::readAttr() {
  _reader.begin_object();
  readKey("type");
  _reader.read_string(&_str);
  readKey("value", &_str2);

  mlir::Attribute attr;

  if (_str == "bool") {
    _reader.read_number(&_uS);
    attr = _builder.getBoolAttr(_uS != 0);
  } else if (_str == "s64") {
    _reader.read_number(&_i64);
    attr = _builder.getI64IntegerAttr(_i64);
  } else if (_str == "f32") {
    _reader.read_number(&_f32);
    attr = _builder.getF32FloatAttr(_f32);
  } else if (_str == "s64[]") {
    _ia64.clear();
    readNumArray(_ia64);
    attr = _builder.getI64ArrayAttr(_ia64);
  } else if (_str == "f32[]") {
    _fa32.clear();
    readNumArray(_fa32);
    attr = _builder.getF32ArrayAttr(_fa32);
  } else if (_str == "string") {
    _reader.read_string(&_str);
    attr = _builder.getStringAttr(_str);
  } else {
    throwErr<std::logic_error>("Unsupported attribute type: ");
  }

  if (_reader.next_object_item(&_str)) {
    throwUnrecognizedKey();
  }

  return attr;
}

mlir::Type JsonParser::readTensorType() {
  GetTypeFn getTypeFn = nullptr;
  _ia64.clear();
  _reader.begin_object();

  while (_reader.next_object_item(&_str)) {
    if (_str == "id") {
      _reader.read_number(&_uS);
    } else if (_str == "dtype") {
      _reader.read_string(&_str);
      auto fn = _dtypes.find(_str);
      if (fn == _dtypes.end()) {
        throwErr<std::logic_error>("Unsupported dtype: ");
      }
      getTypeFn = fn->second;
    } else if (_str == "shape") {
      readNumArray(_ia64);
    } else if (_str == "stride") {
      _ia642.clear();
      readNumArray(_ia642);
      if ((_ia642.size() > 1) ||
          ((_ia642.size() == 1) &&
           (_ia642[0] != std::numeric_limits<int64_t>::min()))) {
        // TODO: Add support for strides
        throwErr<std::logic_error>("Unsupported stride value: ");
      }
    } else if (_str == "layout_type") {
      _reader.read_string(&_str);
      if ((_str != "undef") && (_str != "any")) {
        throwErr<std::logic_error>("Unsupported layout_type: ");
      }
    } else if (_str == "property_type") {
      _reader.read_string(&_str);
      if ((_str != "undef") && (_str != "constant")) {
        throwErr<std::logic_error>("Unsupported property_type: ");
      }
    } else {
      throwUnrecognizedKey();
    }
  }

  if (getTypeFn == nullptr) {
    _str.clear();
    throwErr<std::invalid_argument>("dtype is not specified");
  }

  if ((_ia64.size() == 1) &&
      (_ia64[0] == std::numeric_limits<int64_t>::min())) {
    return mlir::UnrankedTensorType::get(getTypeFn(_builder));
  }

  return mlir::RankedTensorType::get(_ia64, getTypeFn(_builder));
}
