//===-- MicrokernelOps.cpp - microkernel dialect ops ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Microkernel/MicrokernelOps.h"
#include "gc/Dialect/Microkernel/MicrokernelDialect.h"
#include <mlir/IR/TypeUtilities.h>

#define GET_OP_CLASSES
#include "gc/Dialect/Microkernel/MicrokernelOps.cpp.inc"

#include <llvm/Support/Debug.h>

namespace mlir {

namespace microkernel {

constexpr std::string_view INPUTS = "inputs";
constexpr std::string_view DATA_TYPE = "data_type";
constexpr std::string_view FLAGS_NAME = "flags";

template <typename OpTy>
static void printInputImpl(OpAsmPrinter &printer, OpTy op) {
  printer << " [" << op.getInputs() << ']';
}

template <typename AttrTy>
static void printFlagsImpl(OpAsmPrinter &printer,
                           const std::function<ArrayAttr()> &fn,
                           const std::string_view &flagsName) {
  printer << " " << flagsName << " = (";
  llvm::interleaveComma(fn(), printer, [&](auto &flag) {
    printer << stringifyEnum(cast<AttrTy>(flag).getValue());
  });
  printer << ") ";
}

template <typename OpTy>
static void printDataTypeImpl(OpAsmPrinter &printer, OpTy op) {
  printer << DATA_TYPE << " = (";
  auto dataTypes = op.getDataType();
  for (size_t idx = 0; idx < dataTypes.size(); idx++) {
    printer.printAttribute(dataTypes[idx]);
    if (idx != dataTypes.size() - 1) {
      printer << ", ";
    }
  }
  printer << ") ";
}

template <typename EnumClass>
static ParseResult parseEnum(EnumClass &value, OpAsmParser &parser) {
  StringRef flag;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&flag))
    return failure();
  auto flagAttr = symbolizeEnum<EnumClass>(flag);
  if (!flagAttr)
    return parser.emitError(loc, "invalid enum ") << flag;
  value = *flagAttr;
  return success();
}

static ParseResult parseOperandImpl(OpAsmParser &parser,
                                    OperationState &result) {
  DenseI64ArrayAttr kindAttr;
  if (parser.parseCustomAttributeWithFallback(kindAttr, Type{}, INPUTS,
                                              result.attributes)) {
    return failure();
  }
  auto &builder = parser.getBuilder();
  result.addTypes(builder.getIntegerType(64));
  return success();
}

static ParseResult parseDataTypeImpl(OpAsmParser &parser,
                                     OperationState &result) {
  auto &builder = parser.getBuilder();
  if (parser.parseKeyword(DATA_TYPE) || parser.parseEqual() ||
      parser.parseLParen())
    return failure();
  SmallVector<Attribute, 2> dataTypes;
  auto parseTypeAttr = [&]() -> ParseResult {
    Attribute dataType;
    if (parser.parseAttribute(dataType))
      return failure();
    if (!isa<TypeAttr>(dataType))
      return failure();
    dataTypes.push_back(dataType);
    return success();
  };
  if (parser.parseCommaSeparatedList(parseTypeAttr) || parser.parseRParen())
    return failure();

  result.addAttribute(DATA_TYPE, builder.getArrayAttr(dataTypes));
  return success();
}

template <typename FLAGS>
static ParseResult parseFlagsImpl(OpAsmParser &parser, OperationState &result,
                                  const std::string_view &flagsName) {
  auto &builder = parser.getBuilder();
  if (parser.parseKeyword(flagsName) || parser.parseEqual() ||
      parser.parseLParen())
    return failure();

  SmallVector<Attribute, 4> flags;
  auto parseFlags = [&]() -> ParseResult {
    FLAGS flag;
    if (parseEnum<FLAGS>(flag, parser))
      return failure();
    flags.push_back(builder.getI64IntegerAttr(static_cast<int64_t>(flag)));
    return success();
  };
  if (parser.parseCommaSeparatedList(parseFlags) || parser.parseRParen())
    return failure();
  result.addAttribute(flagsName, builder.getArrayAttr(flags));
  return success();
}

template <typename FLAGS>
static LogicalResult
verifyUniquenessAndConsistency(ArrayAttr flags, Operation *op,
                               const std::string_view &flagsName) {
  SmallVector<int64_t> flagsAsInt;
  for (auto flag : flags)
    flagsAsInt.push_back(cast<IntegerAttr>(flag).getInt());

  // check uniqueness
  std::sort(flagsAsInt.begin(), flagsAsInt.end());
  auto *it = std::unique(flagsAsInt.begin(), flagsAsInt.end());
  if (it != flagsAsInt.end())
    return op->emitOpError() << "expected " << flagsName << " to be unique";
  // none flag conflicts with all the others
  if (llvm::is_contained(flagsAsInt, static_cast<int64_t>(FLAGS::NONE)) &&
      flagsAsInt.size() != 1) {
    return op->emitOpError()
           << "'none' " << flagsName << " conflicts with others";
  }
  return success();
}

/////////////////////////////////////////////////////
// Start of BrgemmDispatchOp

void BrgemmDispatchOp::print(OpAsmPrinter &printer) {
  printInputImpl<BrgemmDispatchOp>(printer, *this);
  auto getOpFlags = [this]() -> ArrayAttr { return this->getFlags(); };
  printFlagsImpl<BrgemmFlagsAttr>(printer, getOpFlags, FLAGS_NAME);
  printDataTypeImpl<BrgemmDispatchOp>(printer, *this);
}

ParseResult BrgemmDispatchOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  if (failed(parseOperandImpl(parser, result)) ||
      failed(parseFlagsImpl<BrgemmFlags>(parser, result, FLAGS_NAME)))
    return failure();
  if (failed(parseDataTypeImpl(parser, result)))
    return failure();
  return parser.parseOptionalAttrDict(result.attributes);
}

static LogicalResult verifyBrgemmDataTypes(ArrayAttr dtypes,
                                           BrgemmDispatchOp op) {
  if (dtypes.size() != 2) {
    return op->emitOpError() << "data types size should be 2";
  }

  auto context = op.getContext();

#define FTAttr(t) TypeAttr::get(FloatType::get##t(context))
#define ITAttr(s, w) TypeAttr::get(IntegerType::get(context, w, IntegerType::s))
  SmallVector<std::pair<TypeAttr, TypeAttr>> validDataTypes = {
      {FTAttr(F32), FTAttr(F32)},
      {FTAttr(BF16), FTAttr(BF16)},
      {ITAttr(Unsigned, 8), ITAttr(Signed, 8)},
      {ITAttr(Signed, 8), ITAttr(Unsigned, 8)},
      {ITAttr(Unsigned, 8), ITAttr(Unsigned, 8)},
      {ITAttr(Signed, 8), ITAttr(Signed, 8)}};
#undef FTAttr
#undef ITAttr
  if (!llvm::any_of(validDataTypes,
                    [=](std::pair<TypeAttr, TypeAttr> type_pair) {
                      return type_pair.first == dtypes[0] ||
                             type_pair.second == dtypes[1];
                    })) {
    return op->emitOpError() << "invalid data type pair";
  }

  return success();
}

static LogicalResult verifyBrgemmFlags(ArrayAttr flags, BrgemmDispatchOp op,
                                       const std::string_view &flagsName) {
  // Verify flags.
  if (failed(verifyUniquenessAndConsistency<BrgemmFlags>(flags, op, flagsName)))
    return failure();

  bool strideSet = false;
  bool listSet = false;
  for (auto flag : flags) {
    if (cast<BrgemmFlagsAttr>(flag).getValue() == BrgemmFlags::STRIDE) {
      strideSet = true;
    }
    if (cast<BrgemmFlagsAttr>(flag).getValue() == BrgemmFlags::LIST) {
      listSet = true;
    }
  }
  // VNNI flags must be specified only for bf16 type
  if (strideSet && listSet) {
    return op->emitOpError()
           << "stride and addr flags conflict with each other";
  }

  return success();
}

LogicalResult BrgemmDispatchOp::verify() {
  BrgemmDispatchOp &op = *this;
  // 'inputs' = [m, n, k, lda, ldb, ldc, stride_a, stride_b] for BRGEMM.
  size_t expected = 8;
  size_t numInputs = op.getInputs().size();
  if (numInputs != expected) {
    return op.emitOpError()
           << "expect " << expected << " args but got: " << numInputs;
  }
  // Verify data types
  if (failed(verifyBrgemmDataTypes(op.getDataType(), op))) {
    return failure();
  }

  // Verify leading dims.
  ArrayRef<int64_t> inputs = op.getInputs();
  int64_t n = inputs[1];
  int64_t k = inputs[2];
  int64_t lda = inputs[3];
  int64_t ldb = inputs[4];
  int64_t ldc = inputs[5];
  if (lda < k)
    return op.emitOpError() << "expect lda to be >= of dimension k\n";
  if (ldb < n)
    return op.emitOpError() << "expect ldb to be >= of dimension n\n";
  if (ldc < n)
    return op.emitOpError() << "expect ldc to be >= of dimension n\n";

  // Verify dispatch flags.
  return verifyBrgemmFlags(op.getFlags(), op, FLAGS_NAME);
}

/////////////////////////////////////////////////////
// Start of BrgemmOp

// TODO(haixin): could use compiler-wide VNNI utils?
static bool isInVnniLayout(MemRefType memref) {
  if (!memref.getElementType().isBF16() &&
      !memref.getElementType().isSignedInteger(8) &&
      !memref.getElementType().isUnsignedInteger(8)) {
    return false;
  }

  auto blockingFactor = 0;
  if (memref.getElementType().isBF16()) {
    blockingFactor = 2;
  } else if (memref.getElementType().isSignedInteger(8) ||
             memref.getElementType().isUnsignedInteger(8)) {
    blockingFactor = 4;
  }
  return memref.getShape().back() == blockingFactor;
}

static bool isTypeCompatible(Type outType, Type operandAType,
                             Type operandBType) {
  if (!outType.isF32() && !outType.isSignedInteger(32)) {
    return false;
  }
  if (outType.isF32()) {
    if (!(operandAType.isF32() && operandBType.isF32()) &&
        !(operandAType.isBF16() && operandBType.isBF16())) {
      return false;
    }
  }
  if (outType.isSignedInteger(32)) {
    if (!(operandAType.isSignedInteger(8) ||
          operandAType.isUnsignedInteger(8)) &&
        (operandBType.isSignedInteger(8) ||
         operandBType.isUnsignedInteger(8))) {
      return false;
    }
  }
  return true;
}

LogicalResult BrgemmOp::verify() {
  BrgemmOp &brgemmOp = *this;

  SmallVector<Value> inputs = brgemmOp.getInputs();
  // inputs for BRGEMM: kernel id, A memref, B memref, C memref, batch_size,
  // addr_len
  if (inputs.size() != 6) {
    return brgemmOp.emitOpError() << "expect 6"
                                  << " inputs but got " << inputs.size();
  }
  // Verify the dispatch to be an i64.
  Value dispatch = brgemmOp.getDispatch();
  if (!dispatch.getType().isInteger(64)) {
    return brgemmOp.emitOpError()
           << "expect an i64 but got " << dispatch.getType()
           << " for operand 0 (dispatch)";
  }

  // Verify the compatibility of memref types
  SmallVector<Value> memrefOperands = {
      brgemmOp.getOperandA(), brgemmOp.getOperandB(), brgemmOp.getOutput()};
  SmallVector<Type> typeOperands = {
      getElementTypeOrSelf(memrefOperands[0].getType()),
      getElementTypeOrSelf(memrefOperands[1].getType()),
      getElementTypeOrSelf(memrefOperands[2].getType())};
  if (!isTypeCompatible(typeOperands[2], typeOperands[0], typeOperands[1])) {
    return brgemmOp.emitOpError()
           << "operands types: " << typeOperands[0] << " X " << typeOperands[1]
           << " -> " << typeOperands[2] << " are imcompatible";
  }

  // Verify the rank of the shaped operands.
  for (size_t idx = 0; idx < memrefOperands.size(); idx++) {
    size_t actualIdx = idx + 1 /*skip dispatch*/;
    auto memref = dyn_cast<MemRefType>(memrefOperands[idx].getType());
    // Output memref. Must be of rank 2.
    if (idx == 2 && memref.getRank() != 2) {
      return brgemmOp.emitOpError()
             << "expect a 2d layout for operand: " << actualIdx;
    }
    // Input A memref. Must be of rank 3.
    if (idx == 0 && memref.getRank() != 3) {
      return brgemmOp.emitOpError()
             << "expect a 3d memref for operand: " << actualIdx;
    }
    // Input B memref. Must be in VNNI layout with rank 4 for non-F32.
    if (idx == 1) {
      auto dtype_B = typeOperands[idx];
      if (!dtype_B.isF32()) {
        if (memref.getRank() != 4 && !isInVnniLayout(memref)) {
          return brgemmOp.emitOpError()
                 << "expect a 4d VNNI memref for non-F32 operand: "
                 << actualIdx;
        }
      } else {
        if (memref.getRank() != 3) {
          return brgemmOp.emitOpError()
                 << "expect a 3d memref for F32 operand: " << actualIdx;
        }
      }
    }
  }

  // Verify the batch and addrLen to be i64.
  Value batch = brgemmOp.getBatch();
  if (!batch.getType().isInteger(64)) {
    return brgemmOp.emitOpError() << "expect an i64 but got " << batch.getType()
                                  << " for operand 4 (batch)";
  }
  Value addrLen = brgemmOp.getAddrLen();
  if (!addrLen.getType().isInteger(64)) {
    return brgemmOp.emitOpError()
           << "expect an i64 but got " << addrLen.getType()
           << " for operand 5 (addrLen)";
  }
  return success();
}

} // namespace microkernel
} // namespace mlir
