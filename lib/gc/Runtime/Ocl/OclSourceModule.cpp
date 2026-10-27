// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gc/Runtime/OclSourceModule.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;

namespace mlir::gc::gpu {
namespace {
llvm::Error error(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}
std::string typeName(Type type) {
  if (type.isF32()) return "float";
  if (type.isF16()) return "gc_half";
  if (type.isF64()) return "double";
  if (type.isIndex()) return "long";
  if (auto i = dyn_cast<IntegerType>(type)) {
    switch (i.getWidth()) {
    case 1:
      return "bool";
    case 8:
      return i.isUnsigned() ? "uchar" : "char";
    case 16:
      return i.isUnsigned() ? "ushort" : "short";
    case 32:
      return i.isUnsigned() ? "uint" : "int";
    case 64:
      return i.isUnsigned() ? "ulong" : "long";
    }
  }
  return {};
}
std::string unsignedTypeName(Type type) {
  auto name = typeName(type);
  return name == "char"    ? "uchar"
         : name == "short" ? "ushort"
         : name == "int"   ? "uint"
         : name == "long"  ? "ulong"
                           : name;
}
std::string literal(Attribute attr) {
  if (auto typed = dyn_cast<TypedAttr>(attr);
      typed && typeName(typed.getType()).empty())
    return {};
  if (auto f = dyn_cast<FloatAttr>(attr)) {
    auto bits = f.getValue().bitcastToAPInt().getZExtValue();
    if (f.getType().isF32())
      return llvm::formatv("as_float((uint){0})", bits).str();
    if (f.getType().isF64())
      return llvm::formatv("as_double((ulong){0}UL)", bits).str();
    if (f.getType().isF16())
      return llvm::formatv("as_gc_half((ushort){0})", bits).str();
  }
  if (auto i = dyn_cast<IntegerAttr>(attr)) {
    llvm::SmallString<32> s;
    i.getValue().toString(s, 10, !i.getType().isUnsignedInteger());
    return "(" + typeName(i.getType()) + ")" + s.str().str();
  }
  return {};
}
struct SizeExpr {
  std::string code;
  std::function<std::optional<int64_t>(ArrayRef<int64_t>)> evaluate;

  SizeExpr(int64_t value = 0)
      : code(std::to_string(value)),
        evaluate([value](ArrayRef<int64_t>) { return value; }) {}
  SizeExpr(std::string code,
           std::function<std::optional<int64_t>(ArrayRef<int64_t>)> evaluate)
      : code(std::move(code)), evaluate(std::move(evaluate)) {}
};

SizeExpr sizeBinary(SizeExpr lhs, SizeExpr rhs, char op) {
  auto evaluate = [lhs, rhs,
                   op](ArrayRef<int64_t> sizes) -> std::optional<int64_t> {
    auto a = lhs.evaluate(sizes), b = rhs.evaluate(sizes);
    if (!a || !b) return std::nullopt;
    switch (op) {
    case '+':
      return llvm::checkedAdd(*a, *b);
    case '-':
      return llvm::checkedSub(*a, *b);
    case '*':
      return llvm::checkedMul(*a, *b);
    case '/':
    case '%':
      if (!*b || (*a == std::numeric_limits<int64_t>::min() && *b == -1))
        return std::nullopt;
      return op == '/' ? *a / *b : *a % *b;
    }
    return std::nullopt;
  };
  if (auto constant = evaluate({})) return SizeExpr(*constant);
  if (op == '*' && lhs.code == "1") return rhs;
  if (op == '*' && rhs.code == "1") return lhs;
  if (op == '+' && lhs.code == "0") return rhs;
  if (op == '+' && rhs.code == "0") return lhs;
  return SizeExpr("(" + lhs.code + op + rhs.code + ")", evaluate);
}

struct Buffer {
  Type type;
  SizeExpr bytes;
  std::vector<char> initial;
};
struct View {
  size_t slot;
  MemRefType type;
  SizeExpr offset;
  SmallVector<SizeExpr> sizes;
  SmallVector<SizeExpr> strides;
};
struct Kernel {
  std::string name;
  std::string body;
  SizeExpr workItems;
  SmallVector<size_t> slots;
};

class SourceBuilder {
public:
  SmallVector<Buffer> buffers;
  SmallVector<Kernel> kernels;
  SmallVector<MemRefType> argTypes;
  SmallVector<std::pair<size_t, size_t>> dynamicDims;

  LogicalResult build(ModuleOp module, bool fp16) {
    supportsFp16 = fp16;
    auto funcs = module.getOps<func::FuncOp>();
    auto it = funcs.begin();
    if (it == funcs.end())
      return module.emitError("Missing OpenCL entry function");
    auto fn = *it++;
    if (it != funcs.end() || fn.getBody().getBlocks().size() != 1 ||
        fn.getNumResults())
      return fn.emitError(
          "OpenCL source requires one function with buffer arguments");
    for (Value arg : fn.getArguments()) {
      auto type = dyn_cast<MemRefType>(arg.getType());
      if (!type || !type.getLayout().isIdentity())
        return fn.emitError("OpenCL source requires contiguous memrefs");
      SmallVector<SizeExpr> sizes;
      for (int64_t i = 0; i < type.getRank(); ++i) {
        if (!type.isDynamicDim(i)) continue;
        size_t index = dynamicDims.size();
        dynamicDims.emplace_back(argTypes.size(), i);
        sizes.emplace_back(
            "s" + std::to_string(index),
            [index](ArrayRef<int64_t> dims) -> std::optional<int64_t> {
              if (index >= dims.size()) return std::nullopt;
              return dims[index];
            });
      }
      if (failed(addBuffer(arg, type, {}, sizes))) return failure();
      argTypes.push_back(type);
    }
    for (Operation &op : fn.getBody().front()) {
      if (auto c = dyn_cast<arith::ConstantOp>(op)) {
        auto value = literal(c.getValue());
        if (value.empty()) return c.emitError("Unsupported OpenCL constant");
        scalars[c] = value;
        if (auto attr = dyn_cast<IntegerAttr>(c.getValue()))
          shapeValues[c] = SizeExpr(attr.getInt());
      } else if (auto a = dyn_cast<memref::AllocOp>(op)) {
        SmallVector<SizeExpr> sizes;
        if (failed(dynamicSizes(a.getDynamicSizes(), sizes)) ||
            failed(addBuffer(a, a.getType(), {}, sizes)))
          return failure();
      } else if (auto a = dyn_cast<memref::AllocaOp>(op)) {
        SmallVector<SizeExpr> sizes;
        if (failed(dynamicSizes(a.getDynamicSizes(), sizes)) ||
            failed(addBuffer(a, a.getType(), {}, sizes)))
          return failure();
      } else if (auto d = dyn_cast<memref::DimOp>(op)) {
        auto it = views.find(d.getSource());
        auto index = d.getConstantIndex();
        if (it == views.end() || !index || *index < 0 ||
            *index >= static_cast<int64_t>(it->second.sizes.size()))
          return d.emitError("Unsupported OpenCL dimension query");
        shapeValues[d] = it->second.sizes[*index];
        scalars[d] = shapeValues[d].code;
      } else if (op.getName().getDialectNamespace() == "arith") {
        if (failed(shapeScalar(op))) return failure();
      } else if (auto g = dyn_cast<memref::GetGlobalOp>(op)) {
        auto global = module.lookupSymbol<memref::GlobalOp>(g.getName());
        auto init = global ? global.getInitialValue() : std::nullopt;
        auto dense =
            init ? dyn_cast<DenseElementsAttr>(*init) : DenseElementsAttr{};
        if (!dense || failed(addBuffer(g, g.getType(), dense)))
          return g.emitError("Unsupported OpenCL global");
      } else if (auto c = dyn_cast<memref::CastOp>(op)) {
        if (failed(alias(c, c.getSource(), cast<MemRefType>(c.getType()))))
          return failure();
      } else if (auto c = dyn_cast<memref::CollapseShapeOp>(op)) {
        auto it = views.find(c.getSrc());
        if (it == views.end()) return failure();
        SmallVector<SizeExpr> sizes;
        for (auto group : c.getReassociationIndices()) {
          SizeExpr size(1);
          for (int64_t dim : group)
            size = sizeBinary(size, it->second.sizes[dim], '*');
          sizes.push_back(size);
        }
        if (failed(reshape(c, c.getSrc(), c.getResultType(), sizes)))
          return failure();
      } else if (auto e = dyn_cast<memref::ExpandShapeOp>(op)) {
        SmallVector<SizeExpr> sizes;
        if (failed(mixedSizes(e.getMixedOutputShape(), sizes)) ||
            failed(reshape(e, e.getSrc(), e.getResultType(), sizes)))
          return failure();
      } else if (auto s = dyn_cast<memref::SubViewOp>(op)) {
        if (failed(subview(s))) return failure();
      } else if (auto l = dyn_cast<linalg::GenericOp>(op)) {
        if (failed(kernel(l))) return failure();
      } else if (auto c = dyn_cast<memref::CopyOp>(op)) {
        if (failed(copy(c))) return failure();
      } else if (!isa<func::ReturnOp, memref::DeallocOp>(op)) {
        return op.emitError("Unsupported operation in OpenCL source backend");
      }
    }
    return success();
  }

private:
  bool supportsFp16 = false;
  llvm::DenseMap<Value, View> views;
  llvm::DenseMap<Value, std::string> scalars;
  llvm::DenseMap<Value, SizeExpr> shapeValues;
  unsigned nextValue = 0;

  LogicalResult dynamicSizes(ValueRange values,
                             SmallVectorImpl<SizeExpr> &sizes) {
    for (Value value : values) {
      auto it = shapeValues.find(value);
      if (it == shapeValues.end())
        return emitError(value.getLoc(), "Unsupported OpenCL dynamic size");
      sizes.push_back(it->second);
    }
    return success();
  }
  LogicalResult mixedSizes(ArrayRef<OpFoldResult> values,
                           SmallVectorImpl<SizeExpr> &sizes) {
    for (auto value : values) {
      if (auto attr = dyn_cast<Attribute>(value))
        sizes.emplace_back(cast<IntegerAttr>(attr).getInt());
      else if (failed(dynamicSizes(ValueRange{cast<Value>(value)}, sizes)))
        return failure();
    }
    return success();
  }
  LogicalResult shapeScalar(Operation &op) {
    if (op.getNumResults() != 1 || op.getNumOperands() < 1)
      return op.emitError("Unsupported OpenCL shape operation");
    SmallVector<SizeExpr> operands;
    if (failed(dynamicSizes(op.getOperands(), operands))) return failure();
    auto name = op.getName().getStringRef();
    SizeExpr result;
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op)) result = operands[0];
    else if (operands.size() == 2) {
      char symbol = name == "arith.addi"                             ? '+'
                    : name == "arith.subi"                           ? '-'
                    : name == "arith.muli"                           ? '*'
                    : name == "arith.divsi" || name == "arith.divui" ? '/'
                    : name == "arith.remsi" || name == "arith.remui" ? '%'
                                                                     : 0;
      if (!symbol) return op.emitError("Unsupported OpenCL shape operation");
      result = sizeBinary(operands[0], operands[1], symbol);
    } else return op.emitError("Unsupported OpenCL shape operation");
    shapeValues[op.getResult(0)] = result;
    scalars[op.getResult(0)] = result.code;
    return success();
  }
  LogicalResult addBuffer(Value value, MemRefType type, DenseElementsAttr init,
                          ArrayRef<SizeExpr> dynamic = {}) {
    auto name = typeName(type.getElementType());
    if (!type.getLayout().isIdentity() || name.empty() ||
        dynamic.size() != static_cast<size_t>(type.getNumDynamicDims()))
      return emitError(value.getLoc(), "Unsupported OpenCL buffer type");
    unsigned width =
        type.getElementType().isIndex() ? 64 : type.getElementTypeBitWidth();
    size_t elementBytes = std::max(1u, width / 8);
    SmallVector<SizeExpr> sizes, strides(type.getRank());
    size_t dynamicIndex = 0;
    for (int64_t size : type.getShape())
      sizes.push_back(ShapedType::isDynamic(size) ? dynamic[dynamicIndex++]
                                                  : SizeExpr(size));
    SizeExpr stride(1);
    for (int64_t i = type.getRank(); i-- > 0;) {
      strides[i] = stride;
      stride = sizeBinary(stride, sizes[i], '*');
    }
    auto bytes = sizeBinary(stride, SizeExpr(elementBytes), '*');
    Buffer buffer{type.getElementType(), bytes, {}};
    if (init) {
      auto count = bytes.evaluate({});
      if (!count || *count < 0)
        return emitError(value.getLoc(), "OpenCL buffer size overflow");
      buffer.initial.resize(*count);
      size_t offset = 0;
      for (Attribute attr : init.getValues<Attribute>()) {
        uint64_t bits = isa<FloatAttr>(attr)
                            ? cast<FloatAttr>(attr)
                                  .getValue()
                                  .bitcastToAPInt()
                                  .getZExtValue()
                            : cast<IntegerAttr>(attr).getValue().getZExtValue();
        std::memcpy(buffer.initial.data() + offset, &bits, elementBytes);
        offset += elementBytes;
      }
    }
    views[value] = View{buffers.size(), type, SizeExpr(0), sizes, strides};
    buffers.push_back(std::move(buffer));
    return success();
  }
  LogicalResult alias(Value value, Value source, MemRefType type) {
    auto it = views.find(source);
    if (it == views.end())
      return emitError(value.getLoc(), "Unsupported OpenCL buffer alias");
    views[value] = View{it->second.slot, type, it->second.offset,
                        it->second.sizes, it->second.strides};
    return success();
  }
  LogicalResult reshape(Value value, Value source, MemRefType type,
                        ArrayRef<SizeExpr> sizes) {
    auto it = views.find(source);
    if (it == views.end()) return failure();
    auto view = it->second;
    SizeExpr stride(1);
    for (int64_t i = view.type.getRank(); i-- > 0;) {
      if (view.strides[i].code != stride.code)
        return emitError(value.getLoc(),
                         "OpenCL reshape requires contiguous storage");
      stride = sizeBinary(stride, view.sizes[i], '*');
    }
    SmallVector<SizeExpr> strides(type.getRank());
    stride = SizeExpr(1);
    for (int64_t i = type.getRank(); i-- > 0;) {
      strides[i] = stride;
      stride = sizeBinary(stride, sizes[i], '*');
    }
    views[value] = View{view.slot, type, view.offset,
                        SmallVector<SizeExpr>(sizes), strides};
    return success();
  }
  LogicalResult subview(memref::SubViewOp op) {
    auto it = views.find(op.getSource());
    if (it == views.end()) return failure();
    auto view = it->second;
    SmallVector<SizeExpr> offsets, sizes, strides;
    if (failed(mixedSizes(op.getMixedOffsets(), offsets)) ||
        failed(mixedSizes(op.getMixedSizes(), sizes)) ||
        failed(mixedSizes(op.getMixedStrides(), strides)))
      return failure();
    SmallVector<SizeExpr> newStrides, newSizes;
    auto dropped = op.getDroppedDims();
    for (size_t i = 0; i < offsets.size(); ++i) {
      view.offset = sizeBinary(
          view.offset, sizeBinary(offsets[i], view.strides[i], '*'), '+');
      if (!dropped.test(i)) {
        newStrides.push_back(sizeBinary(view.strides[i], strides[i], '*'));
        newSizes.push_back(sizes[i]);
      }
    }
    views[op] =
        View{view.slot, op.getType(), view.offset, newSizes, newStrides};
    return success();
  }
  std::string affine(AffineExpr expr) {
    if (auto d = dyn_cast<AffineDimExpr>(expr))
      return "i" + std::to_string(d.getPosition());
    if (auto c = dyn_cast<AffineConstantExpr>(expr))
      return std::to_string(c.getValue());
    if (auto b = dyn_cast<AffineBinaryOpExpr>(expr)) {
      auto lhs = affine(b.getLHS()), rhs = affine(b.getRHS());
      if (lhs.empty() || rhs.empty()) return {};
      switch (b.getKind()) {
      case AffineExprKind::Add:
        return "(" + lhs + "+" + rhs + ")";
      case AffineExprKind::Mul:
        return "(" + lhs + "*" + rhs + ")";
      case AffineExprKind::FloorDiv:
        return "gc_floor_div(" + lhs + "," + rhs + ")";
      case AffineExprKind::CeilDiv:
        return "(-gc_floor_div(-(" + lhs + ")," + rhs + "))";
      case AffineExprKind::Mod:
        return "(" + lhs + "-gc_floor_div(" + lhs + "," + rhs + ")*" + rhs +
               ")";
      default:
        break;
      }
    }
    return {};
  }
  std::string access(Value value, ArrayRef<std::string> indices, Kernel &k) {
    auto it = views.find(value);
    if (it == views.end() || indices.size() != it->second.strides.size())
      return {};
    auto &view = it->second;
    if (!llvm::is_contained(k.slots, view.slot)) k.slots.push_back(view.slot);
    std::string index = view.offset.code;
    for (size_t i = 0; i < indices.size(); ++i)
      index += "+(" + indices[i] + ")*" + view.strides[i].code;
    return "b" + std::to_string(view.slot) + "[" + index + "]";
  }
  std::string mappedAccess(Value value, AffineMap map, Kernel &k) {
    SmallVector<std::string> indices;
    for (auto expr : map.getResults()) {
      auto index = affine(expr);
      if (index.empty()) return {};
      indices.push_back(index);
    }
    return access(value, indices, k);
  }
  LogicalResult scalar(Operation &op,
                       llvm::DenseMap<Value, std::string> &values, Kernel &k) {
    if (op.getNumResults() != 1)
      return op.emitError("Unsupported OpenCL scalar operation");
    if (!supportsFp16 &&
        !isa<arith::ConstantOp, memref::LoadOp, arith::BitcastOp,
             arith::SelectOp>(op) &&
        (op.getResult(0).getType().isF16() ||
         llvm::any_of(op.getOperandTypes(),
                      [](Type type) { return type.isF16(); })))
      return op.emitError("OpenCL device does not support fp16 arithmetic");
    std::string expression;
    if (auto c = dyn_cast<arith::ConstantOp>(op))
      expression = literal(c.getValue());
    else if (auto i = dyn_cast<linalg::IndexOp>(op))
      expression = "i" + std::to_string(i.getDim());
    else if (auto l = dyn_cast<memref::LoadOp>(op)) {
      SmallVector<std::string> indices;
      for (Value index : l.getIndices()) {
        auto it = values.find(index);
        if (it == values.end()) return l.emitError("Missing OpenCL load index");
        indices.push_back(it->second);
      }
      expression = access(l.getMemRef(), indices, k);
    } else {
      SmallVector<std::string> operands;
      for (Value operand : op.getOperands()) {
        auto it = values.find(operand);
        if (it == values.end())
          return op.emitError("Missing OpenCL scalar operand");
        operands.push_back(it->second);
      }
      auto name = op.getName().getStringRef();
      auto binary = [&](StringRef symbol) {
        if (operands.size() == 2)
          expression = "(" + operands[0] + symbol.str() + operands[1] + ")";
      };
      if (name == "arith.addi" || name == "arith.subi" ||
          name == "arith.muli") {
        auto resultType = op.getResult(0).getType();
        auto bits = resultType.isIndex()
                        ? 64
                        : cast<IntegerType>(resultType).getWidth();
        auto unsignedType = bits == 64 ? "ulong" : "uint";
        auto symbol = name == "arith.addi"   ? "+"
                      : name == "arith.subi" ? "-"
                                             : "*";
        auto resultName = typeName(resultType);
        auto calculation = "((" + std::string(unsignedType) + ")" +
                           operands[0] + symbol + "(" + unsignedType + ")" +
                           operands[1] + ")";
        expression = bits == 1 ? "((" + calculation + "&1u)!=0)"
                               : "((" + resultName + ")" + calculation + ")";
      } else if (name == "arith.addf") binary("+");
      else if (name == "arith.subf") binary("-");
      else if (name == "arith.mulf") binary("*");
      else if (name == "arith.divf" || name == "arith.divsi") binary("/");
      else if (name == "arith.remsi") binary("%");
      else if (name == "arith.divui" || name == "arith.remui") {
        auto type = unsignedTypeName(op.getOperand(0).getType());
        expression = "(((" + type + ")" + operands[0] +
                     (name == "arith.divui" ? "/(" : "%(") + type + ")" +
                     operands[1] + ")";
      } else if (name == "arith.andi") binary("&");
      else if (name == "arith.ori") binary("|");
      else if (name == "arith.xori") binary("^");
      else if (name == "arith.negf") expression = "(-" + operands[0] + ")";
      else if (isa<arith::ExtSIOp, arith::SIToFPOp, arith::IndexCastOp>(op) &&
               op.getOperand(0).getType().isInteger(1))
        expression = "((" + typeName(op.getResult(0).getType()) + ")(-((int)" +
                     operands[0] + ")))";
      else if (isa<arith::UIToFPOp, arith::ExtUIOp, arith::IndexCastUIOp>(op))
        expression = "((" + typeName(op.getResult(0).getType()) + ")((" +
                     unsignedTypeName(op.getOperand(0).getType()) + ")" +
                     operands[0] + "))";
      else if (isa<arith::FPToUIOp>(op))
        expression = "((" + unsignedTypeName(op.getResult(0).getType()) + ")" +
                     operands[0] + ")";
      else if (isa<arith::BitcastOp>(op))
        expression = "as_" + typeName(op.getResult(0).getType()) + "(" +
                     operands[0] + ")";
      else if (isa<arith::SelectOp>(op))
        expression =
            "(" + operands[0] + "?" + operands[1] + ":" + operands[2] + ")";
      else if (isa<arith::IndexCastOp, arith::SIToFPOp, arith::FPToSIOp,
                   arith::ExtSIOp, arith::TruncIOp, arith::ExtFOp,
                   arith::TruncFOp>(op))
        expression = "((" + typeName(op.getResult(0).getType()) + ")" +
                     operands[0] + ")";
      else if (name == "arith.maximumf" || name == "arith.minimumf") {
        std::string fn = name == "arith.maximumf" ? "fmax" : "fmin";
        expression = "((isnan(" + operands[0] + ")||isnan(" + operands[1] +
                     "))?NAN:" + fn + "(" + operands[0] + "," + operands[1] +
                     "))";
      } else if (name == "arith.maxnumf" || name == "arith.minnumf" ||
                 name == "arith.maxsi" || name == "arith.minsi") {
        std::string fn = name == "arith.maxnumf"   ? "fmax"
                         : name == "arith.minnumf" ? "fmin"
                         : name == "arith.maxsi"   ? "max"
                                                   : "min";
        expression = fn + "(" + operands[0] + "," + operands[1] + ")";
      } else if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
        auto pred = cmp.getPredicate();
        static const char *symbols[] = {"==", "!=", "<",  "<=", ">",
                                        ">=", "<",  "<=", ">",  ">="};
        if (static_cast<unsigned>(pred) >= 6) {
          auto type = unsignedTypeName(op.getOperand(0).getType());
          for (auto &operand : operands)
            operand = "((" + type + ")" + operand + ")";
        }
        binary(symbols[static_cast<unsigned>(pred)]);
      } else if (auto cmp = dyn_cast<arith::CmpFOp>(op)) {
        auto pred = cmp.getPredicate();
        std::string a = operands[0], b = operands[1];
        static const char *symbols[] = {
            "",   "==", ">",  ">=", "<",  "<=", "!=", "",
            "==", ">",  ">=", "<",  "<=", "!=", "",   ""};
        if (pred == arith::CmpFPredicate::AlwaysFalse) expression = "false";
        else if (pred == arith::CmpFPredicate::AlwaysTrue) expression = "true";
        else if (pred == arith::CmpFPredicate::ORD)
          expression = "(!isnan(" + a + ")&&!isnan(" + b + "))";
        else if (pred == arith::CmpFPredicate::UNO)
          expression = "(isnan(" + a + ")||isnan(" + b + "))";
        else {
          bool unordered = static_cast<unsigned>(pred) >= 8;
          expression = "(" + std::string(unordered ? "(isnan(" : "(!isnan(") +
                       a + (unordered ? ")||isnan(" : ")&&!isnan(") + b +
                       (unordered ? "))||(" : "))&&(") + a +
                       symbols[static_cast<unsigned>(pred)] + b + "))";
        }
      } else if (name.starts_with("math.")) {
        auto fn = name.drop_front(5);
        if (fn == "absf") fn = "fabs";
        else if (fn == "absi") fn = "abs";
        else if (fn == "powf") fn = "pow";
        if (fn == "fabs" || fn == "abs" || fn == "exp" || fn == "exp2" ||
            fn == "log" || fn == "log2" || fn == "sqrt" || fn == "rsqrt" ||
            fn == "tanh" || fn == "ceil" || fn == "floor" || fn == "pow" ||
            fn == "fma" || fn == "erf" || fn == "sin" || fn == "cos") {
          expression = fn.str() + "(";
          for (size_t i = 0; i < operands.size(); ++i)
            expression += (i ? "," : "") + operands[i];
          expression += ")";
        }
      }
    }
    auto type = typeName(op.getResult(0).getType());
    if (expression.empty() || type.empty())
      return op.emitError("Unsupported OpenCL scalar operation");
    auto name = "v" + std::to_string(nextValue++);
    k.body += type + " " + name + "=" + expression + ";\n";
    values[op.getResult(0)] = name;
    return success();
  }
  LogicalResult kernel(linalg::GenericOp op) {
    auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    SmallVector<SizeExpr> dims;
    for (Value operand : op.getOperands()) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (type) {
        auto it = views.find(operand);
        if (it == views.end()) return op.emitError("Missing OpenCL buffer");
        llvm::append_range(dims, it->second.sizes);
      }
    }
    auto shapeMap = linalgOp.getShapesToLoopsMap();
    if (!shapeMap) return op.emitError("Cannot infer OpenCL loop bounds");
    SmallVector<SizeExpr> bounds;
    for (auto expr : shapeMap.getResults())
      bounds.push_back(dims[cast<AffineDimExpr>(expr).getPosition()]);
    auto iterators = linalgOp.getIteratorTypesArray();
    Kernel k{"gc_source_" + std::to_string(kernels.size()), {}, 1, {}};
    k.body = "size_t tid=get_global_id(0);\n";
    for (size_t i = bounds.size(); i-- > 0;) {
      if (iterators[i] == utils::IteratorType::parallel) {
        k.workItems = sizeBinary(k.workItems, bounds[i], '*');
        auto divisor = bounds[i].evaluate({});
        auto extent = divisor ? std::to_string(std::max<int64_t>(*divisor, 1))
                              : "max((long)(" + bounds[i].code + "),1L)";
        k.body += "long i" + std::to_string(i) + "=tid%" + extent +
                  ";tid/=" + extent + ";\n";
      }
    }
    for (size_t i = 0; i < bounds.size(); ++i) {
      if (iterators[i] == utils::IteratorType::reduction)
        k.body += "for(long i" + std::to_string(i) + "=0;i" +
                  std::to_string(i) + "<" + bounds[i].code + ";++i" +
                  std::to_string(i) + "){\n";
    }
    llvm::DenseMap<Value, std::string> values = scalars;
    auto maps = op.getIndexingMapsArray();
    auto &body = op.getRegion().front();
    for (size_t i = 0; i < op.getNumOperands(); ++i) {
      Value operand = op.getOperand(i);
      std::string expr = isa<MemRefType>(operand.getType())
                             ? mappedAccess(operand, maps[i], k)
                             : values.lookup(operand);
      if (expr.empty()) return op.emitError("Unsupported OpenCL indexing");
      values[body.getArgument(i)] = expr;
    }
    for (Operation &nested : body.without_terminator())
      if (failed(scalar(nested, values, k))) return failure();
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    for (size_t i = 0; i < yield.getNumOperands(); ++i) {
      auto operandIndex = op.getNumDpsInputs() + i;
      auto dest =
          mappedAccess(op.getOperand(operandIndex), maps[operandIndex], k);
      auto result = values.lookup(yield.getOperand(i));
      if (dest.empty() || result.empty())
        return op.emitError("Missing OpenCL output");
      k.body += dest + "=" + result + ";\n";
    }
    for (auto iterator : iterators)
      if (iterator == utils::IteratorType::reduction) k.body += "}\n";
    kernels.push_back(std::move(k));
    return success();
  }
  LogicalResult copy(memref::CopyOp op) {
    auto type = cast<MemRefType>(op.getSource().getType());
    auto it = views.find(op.getSource());
    if (it == views.end()) return op.emitError("Missing OpenCL copy source");
    Kernel k{"gc_source_" + std::to_string(kernels.size()),
             "size_t tid=get_global_id(0);\n",
             SizeExpr(1),
             {}};
    SmallVector<std::string> indices(type.getRank());
    for (int64_t i = type.getRank(); i-- > 0;) {
      indices[i] = "i" + std::to_string(i);
      auto size = it->second.sizes[i];
      k.workItems = sizeBinary(k.workItems, size, '*');
      auto constant = size.evaluate({});
      auto dim = constant ? std::to_string(std::max<int64_t>(*constant, 1))
                          : "max((long)(" + size.code + "),1L)";
      k.body += "long " + indices[i] + "=tid%" + dim + ";tid/=" + dim + ";\n";
    }
    auto src = access(op.getSource(), indices, k),
         dst = access(op.getTarget(), indices, k);
    if (src.empty() || dst.empty())
      return op.emitError("Unsupported OpenCL copy");
    k.body += dst + "=" + src + ";\n";
    kernels.push_back(std::move(k));
    return success();
  }
};
} // namespace

struct OclSourceModule::Impl {
  cl_context context = nullptr;
  cl_device_id device = nullptr;
  cl_program program = nullptr;
  SmallVector<MemRefType> argTypes;
  SmallVector<std::pair<size_t, size_t>> dynamicDims;
  SmallVector<Buffer> buffers;
  SmallVector<Kernel> kernels;
  ~Impl() {
    if (program) clReleaseProgram(program);
    if (context) clReleaseContext(context);
  }
};

bool useOclSourceBackend(cl_device_id device) {
  (void)device;
#if GC_ENABLE_IGC
  cl_uint vendor = 0;
  return clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor,
                         nullptr) != CL_SUCCESS ||
         vendor != 0x8086;
#else
  return true;
#endif
}

llvm::Expected<OclSourceModule> OclSourceModule::build(ModuleOp module,
                                                       cl_device_id device,
                                                       cl_context context) {
  PassManager pm(module.getContext());
  populateOclSourcePipeline(pm);
  if (failed(pm.run(module)))
    return error("OpenCL source MLIR pipeline failed");
  size_t extSize = 0;
  if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &extSize) !=
      CL_SUCCESS)
    return error("Cannot query OpenCL extensions");
  std::string extensions(extSize, '\0');
  if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extSize, extensions.data(),
                      nullptr) != CL_SUCCESS)
    return error("Cannot query OpenCL extensions");
  const bool fp16 = extensions.find("cl_khr_fp16") != std::string::npos;
  SourceBuilder builder;
  if (failed(builder.build(module, fp16)))
    return error("OpenCL source generation failed");
  auto impl = std::make_shared<Impl>();
  if (clRetainContext(context) != CL_SUCCESS)
    return error("Cannot retain OpenCL context");
  impl->context = context;
  impl->device = device;
  impl->argTypes = std::move(builder.argTypes);
  impl->dynamicDims = std::move(builder.dynamicDims);
  impl->buffers = std::move(builder.buffers);
  impl->kernels = std::move(builder.kernels);
  std::string source;
  source += fp16 ? "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
                   "typedef half gc_half;\n#define as_gc_half as_half\n"
                 : "typedef ushort gc_half;\n#define as_gc_half as_ushort\n";
  if (llvm::any_of(impl->buffers,
                   [](const Buffer &b) { return b.type.isF64(); })) {
    if (extensions.find("cl_khr_fp64") == std::string::npos)
      return error("OpenCL device does not support fp64");
    source += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  }
  source += "#pragma OPENCL FP_CONTRACT OFF\n";
  source += "long gc_floor_div(long a,long b){long q=a/b;long r=a%b;return "
            "q-((r!=0)&&((r<0)!=(b<0)));}\n";
  for (auto &kernel : impl->kernels) {
    source += "__kernel void " + kernel.name + "(";
    for (size_t i = 0; i < kernel.slots.size(); ++i) {
      auto slot = kernel.slots[i];
      source += (i ? "," : "") + std::string("__global ") +
                typeName(impl->buffers[slot].type) + " *b" +
                std::to_string(slot);
    }
    for (size_t i = 0; i < impl->dynamicDims.size(); ++i)
      source += (i || !kernel.slots.empty() ? "," : "") +
                std::string("long s") + std::to_string(i);
    source += "){\n" + kernel.body + "}\n";
  }
  if (impl->kernels.empty()) return OclSourceModule(impl);
  const char *text = source.c_str();
  size_t length = source.size();
  cl_int status;
  impl->program =
      clCreateProgramWithSource(context, 1, &text, &length, &status);
  if (status != CL_SUCCESS)
    return error("clCreateProgramWithSource failed: " + std::to_string(status));
  status = clBuildProgram(impl->program, 1, &device, "-cl-std=CL1.2", nullptr,
                          nullptr);
  if (status != CL_SUCCESS) {
    size_t size = 0;
    clGetProgramBuildInfo(impl->program, device, CL_PROGRAM_BUILD_LOG, 0,
                          nullptr, &size);
    std::string log(size, '\0');
    if (size)
      clGetProgramBuildInfo(impl->program, device, CL_PROGRAM_BUILD_LOG, size,
                            log.data(), nullptr);
    return error("OpenCL source build failed: " + std::to_string(status) +
                 "\n" + log);
  }
  return OclSourceModule(impl);
}

llvm::Error OclSourceModule::invoke(cl_command_queue queue,
                                    llvm::ArrayRef<cl_mem> args,
                                    llvm::ArrayRef<cl_event> waitList,
                                    cl_event *completion) const {
  return invoke(queue, args, {}, waitList, completion);
}

bool OclSourceModule::isStatic() const { return impl->dynamicDims.empty(); }

llvm::Error
OclSourceModule::invoke(cl_command_queue queue, llvm::ArrayRef<cl_mem> args,
                        llvm::ArrayRef<llvm::ArrayRef<int64_t>> shapes,
                        llvm::ArrayRef<cl_event> waitList,
                        cl_event *completion) const {
  const auto argCount = impl->argTypes.size();
  if (args.size() != argCount)
    return error("OpenCL source argument count mismatch");
  if ((!shapes.empty() || !impl->dynamicDims.empty()) &&
      shapes.size() != argCount)
    return error("OpenCL source shape count mismatch");
  for (size_t i = 0; i < shapes.size(); ++i) {
    auto type = impl->argTypes[i];
    if (shapes[i].size() != static_cast<size_t>(type.getRank()))
      return error("OpenCL source argument rank mismatch");
    for (int64_t j = 0; j < type.getRank(); ++j)
      if (shapes[i][j] < 0 ||
          (!type.isDynamicDim(j) && type.getDimSize(j) != shapes[i][j]))
        return error("OpenCL source argument shape mismatch");
  }
  SmallVector<int64_t> dimensions;
  for (auto [arg, dim] : impl->dynamicDims)
    dimensions.push_back(shapes[arg][dim]);
  SmallVector<size_t> bufferBytes, workSizes;
  for (auto &buffer : impl->buffers) {
    auto bytes = buffer.bytes.evaluate(dimensions);
    if (!bytes || *bytes < 0)
      return error("OpenCL source buffer size overflow");
    bufferBytes.push_back(*bytes);
  }
  for (auto &kernel : impl->kernels) {
    auto size = kernel.workItems.evaluate(dimensions);
    if (!size || *size < 0) return error("OpenCL source work size overflow");
    workSizes.push_back(*size);
  }
  cl_context context;
  cl_device_id device;
  if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context), &context,
                            nullptr) != CL_SUCCESS ||
      clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device,
                            nullptr) != CL_SUCCESS ||
      context != impl->context || device != impl->device)
    return error("OpenCL source queue does not match the compiled module");
  for (size_t i = 0; i < args.size(); ++i) {
    size_t bytes = 0;
    cl_context argContext = nullptr;
    if (clGetMemObjectInfo(args[i], CL_MEM_SIZE, sizeof(bytes), &bytes,
                           nullptr) != CL_SUCCESS ||
        clGetMemObjectInfo(args[i], CL_MEM_CONTEXT, sizeof(argContext),
                           &argContext, nullptr) != CL_SUCCESS ||
        bytes < bufferBytes[i] || argContext != impl->context)
      return error(
          "OpenCL source requires buffers with matching size and context");
  }
  struct Invocation {
    SmallVector<cl_mem> owned;
    cl_event event = nullptr;
    ~Invocation() {
      if (event) clReleaseEvent(event);
      for (auto buffer : owned) clReleaseMemObject(buffer);
    }
  } invocation;
  SmallVector<cl_mem> buffers(args.begin(), args.end());
  for (size_t i = argCount; i < impl->buffers.size(); ++i) {
    auto &spec = impl->buffers[i];
    cl_int status;
    auto flags =
        CL_MEM_READ_WRITE | (spec.initial.empty() ? 0 : CL_MEM_COPY_HOST_PTR);
    cl_mem buffer = clCreateBuffer(
        impl->context, flags, std::max<size_t>(bufferBytes[i], 1),
        spec.initial.empty() ? nullptr
                             : const_cast<char *>(spec.initial.data()),
        &status);
    if (status != CL_SUCCESS)
      return error("OpenCL source allocation failed: " +
                   std::to_string(status));
    invocation.owned.push_back(buffer);
    buffers.push_back(buffer);
  }
  for (size_t kernelIndex = 0; kernelIndex < impl->kernels.size();
       ++kernelIndex) {
    auto &spec = impl->kernels[kernelIndex];
    size_t workItems = workSizes[kernelIndex];
    if (!workItems) continue;
    cl_int status;
    auto kernel = clCreateKernel(impl->program, spec.name.c_str(), &status);
    if (status != CL_SUCCESS)
      return error("OpenCL source kernel creation failed: " +
                   std::to_string(status));
    for (size_t i = 0; i < spec.slots.size(); ++i) {
      status =
          clSetKernelArg(kernel, i, sizeof(cl_mem), &buffers[spec.slots[i]]);
      if (status != CL_SUCCESS) break;
    }
    for (size_t i = 0; status == CL_SUCCESS && i < dimensions.size(); ++i)
      status = clSetKernelArg(kernel, spec.slots.size() + i, sizeof(cl_long),
                              &dimensions[i]);
    cl_event next = nullptr;
    if (status == CL_SUCCESS) {
      auto count =
          invocation.event ? 1u : static_cast<cl_uint>(waitList.size());
      auto waits = invocation.event   ? &invocation.event
                   : waitList.empty() ? nullptr
                                      : waitList.data();
      status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &workItems,
                                      nullptr, count, waits, &next);
    }
    clReleaseKernel(kernel);
    if (status != CL_SUCCESS)
      return error("OpenCL source kernel launch failed: " +
                   std::to_string(status));
    if (invocation.event) clReleaseEvent(invocation.event);
    invocation.event = next;
  }
  if (!invocation.event) {
    auto status = clEnqueueMarkerWithWaitList(
        queue, waitList.size(), waitList.empty() ? nullptr : waitList.data(),
        &invocation.event);
    if (status != CL_SUCCESS)
      return error("OpenCL source marker failed: " + std::to_string(status));
  }
  if (completion) {
    *completion = invocation.event;
    invocation.event = nullptr;
  }
  return llvm::Error::success();
}
} // namespace mlir::gc::gpu
