//===-- MicrokernelOps.cpp - microkernel dialect ops ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "gc/Dialect/Microkernel/MicrokernelDialect.h"
#include "gc/Dialect/Microkernel/MicrokernelOps.h"

#define GET_OP_CLASSES
#include "gc/Dialect/Microkernel/MicrokernelOps.cpp.inc"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "microkernel-ops"

using namespace mlir::bufferization;

namespace mlir {

namespace microkernel {

constexpr std::string_view INPUTS_ASM_NAME = "ins";
constexpr std::string_view OUTPUTS_ASM_NAME = "outs";
constexpr std::string_view DATA_TYPE_ASM_NAME = "data_type";
constexpr std::string_view FLAGS_ASM_NAME = "flags";
constexpr std::string_view BATCH_DIMS_ASM_NAME = "batch_dims";
constexpr std::string_view LEADING_DIMS_ASM_NAME = "leading_dims";

constexpr std::string_view INPUTS_ATTR_NAME = "inputs";
constexpr std::string_view BATCH_DIMS_ATTR_NAME = "batchDims";
constexpr std::string_view LEADING_DIMS_ATTR_NAME = "leadingDims";

template <typename AttrTy>
static void printFlagsImpl(OpAsmPrinter &printer,
                           const std::function<ArrayAttr()> &fn,
                           const std::string_view &flagsName) {
  printer << " " << flagsName << "(";
  llvm::interleaveComma(fn(), printer, [&](auto &flag) {
    printer << stringifyEnum(cast<AttrTy>(flag).getValue());
  });
  printer << ") ";
}

template <typename OpTy>
static void printDataTypeImpl(OpAsmPrinter &printer, OpTy op) {
  printer << DATA_TYPE_ASM_NAME << "(";
  auto dataTypes = op.getDataType();
  for (size_t idx = 0; idx < dataTypes.size(); idx++) {
    printer.printAttribute(dataTypes[idx]);
    if (idx != dataTypes.size() - 1)
      printer << ", ";
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

static ParseResult
parseDenseI64ArrayAttrImpl(OpAsmParser &parser, OperationState &result,
                           const std::string_view &attrAsmName,
                           const std::string_view &attrName) {
  auto &builder = parser.getBuilder();
  if (parser.parseKeyword(attrAsmName) || parser.parseLParen())
    return failure();
  SmallVector<int64_t, 2> vals;
  auto parseVal = [&]() -> ParseResult {
    int64_t val;
    if (parser.parseInteger(val))
      return failure();
    vals.push_back(val);
    return success();
  };
  if (parser.parseCommaSeparatedList(parseVal) || parser.parseRParen())
    return failure();

  auto valAttr = builder.getDenseI64ArrayAttr(vals);
  result.addAttribute(attrName, valAttr);
  return success();
}

template <typename AttrType>
static ParseResult parseArrayAttrImpl(OpAsmParser &parser,
                                      OperationState &result,
                                      const std::string_view &attrAsmName,
                                      const std::string_view &attrName) {
  auto &builder = parser.getBuilder();
  if (parser.parseKeyword(attrAsmName) || parser.parseLParen())
    return failure();
  SmallVector<Attribute, 2> attrs;
  auto parseAttr = [&]() -> ParseResult {
    AttrType attr;
    if (parser.parseAttribute(attr))
      return failure();
    attrs.push_back(attr);
    return success();
  };
  if (parser.parseCommaSeparatedList(parseAttr) || parser.parseRParen())
    return failure();

  auto arrayAttr = builder.getArrayAttr(attrs);
  result.addAttribute(attrName, arrayAttr);
  return success();
}

template <typename FLAGS>
static ParseResult parseFlagsImpl(OpAsmParser &parser, OperationState &result,
                                  const std::string_view &flagsName) {
  auto &builder = parser.getBuilder();
  if (parser.parseKeyword(flagsName))
    return failure();

  SmallVector<Attribute, 4> flags;
  auto parseFlags = [&]() -> ParseResult {
    FLAGS flag;
    if (parseEnum<FLAGS>(flag, parser))
      return failure();
    flags.push_back(builder.getI64IntegerAttr(static_cast<int64_t>(flag)));
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseFlags))
    return failure();
  result.addAttribute(flagsName, builder.getArrayAttr(flags));
  return success();
}

static ParseResult parseOperandsImpl(OpAsmParser &parser,
                                     OperationState &result,
                                     const std::string_view &operandsName) {
  SMLoc operandsLoc;
  SmallVector<Type, 1> types;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

  if (parser.parseKeyword(operandsName) || parser.parseLParen())
    return failure();
  operandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operands) || parser.parseColonTypeList(types) ||
      parser.parseRParen())
    return failure();

  if (parser.resolveOperands(operands, types, operandsLoc, result.operands))
    return failure();
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
      flagsAsInt.size() != 1)
    return op->emitOpError()
           << "'none' " << flagsName << " conflicts with others";
  return success();
}

static LogicalResult verifyBrgemmFlags(ArrayAttr flags, Operation *op,
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

static bool isTypeSupported(Type outType, Type operandAType,
                            Type operandBType) {
  if (!outType.isF32() && !outType.isSignedInteger(32))
    return false;

  if (outType.isF32()) {
    if (!(operandAType.isF32() && operandBType.isF32()) &&
        !(operandAType.isBF16() && operandBType.isBF16()))
      return false;
  }
  if (outType.isSignedInteger(32)) {
    if (!(operandAType.isSignedInteger(8) ||
          operandAType.isUnsignedInteger(8)) &&
        (operandBType.isSignedInteger(8) || operandBType.isUnsignedInteger(8)))
      return false;
  }
  return true;
}

// TODO(haixin): could use compiler-wide VNNI utils?
static bool isInVnniLayout(ShapedType type) {
  if (!type.getElementType().isBF16() &&
      !type.getElementType().isSignedInteger(8) &&
      !type.getElementType().isUnsignedInteger(8))
    return false;

  auto blockingFactor = 0;
  if (type.getElementType().isBF16())
    blockingFactor = 2;
  else if (type.getElementType().isSignedInteger(8) ||
           type.getElementType().isUnsignedInteger(8))
    blockingFactor = 4;

  return type.getShape().back() == blockingFactor;
}

/////////////////////////////////////////////////////
// Start of BrgemmOp

ParseResult BrgemmOp::parse(OpAsmParser &parser, OperationState &result) {
  if (failed(parseOperandsImpl(parser, result, INPUTS_ASM_NAME)))
    return failure();
  if (failed(parseOperandsImpl(parser, result, OUTPUTS_ASM_NAME)))
    return failure();

  if (failed(parseDenseI64ArrayAttrImpl(parser, result, BATCH_DIMS_ASM_NAME,
                                        BATCH_DIMS_ATTR_NAME)))
    return failure();
  if (failed(parseDenseI64ArrayAttrImpl(parser, result, LEADING_DIMS_ASM_NAME,
                                        LEADING_DIMS_ATTR_NAME)))
    return failure();

  if (failed(parseFlagsImpl<BrgemmFlags>(parser, result, FLAGS_ASM_NAME)))
    return failure();

  SmallVector<Type, 1> resultTypes;
  if (failed(parser.parseOptionalArrowTypeList(resultTypes)))
    return failure();
  result.addTypes(resultTypes);

  return success();
}

void BrgemmOp::print(OpAsmPrinter &printer) {
  BrgemmOp op = *this;
  ValueRange inputs = op.getInputs();
  Value init = op.getInit();
  printer << " " << INPUTS_ASM_NAME << "(" << inputs << " : "
          << inputs.getTypes() << ")";
  printer << " " << OUTPUTS_ASM_NAME << "(" << init << " : " << init.getType()
          << ")";
  printer << " " << BATCH_DIMS_ASM_NAME << "(" << op.getBatchDims() << ")";
  printer << " " << LEADING_DIMS_ASM_NAME << "(" << op.getLeadingDims() << ")";

  auto getOpFlags = [this]() -> ArrayAttr { return this->getFlags(); };
  printFlagsImpl<BrgemmFlagsAttr>(printer, getOpFlags, FLAGS_ASM_NAME);

  auto resultTypes = op.getResultTypes();
  if (resultTypes.empty())
    return;
  printer.printOptionalArrowTypeList(resultTypes);
}

LogicalResult BrgemmOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void BrgemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (hasPureTensorSemantics())
    return;

  BrgemmOp op = *this;

  for (auto [index, operand] : llvm::enumerate(op.getDpsInputs())) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(
        MemoryEffects::Read::get(), &op->getOpOperand(index), /*stage=*/0,
        /*effectOnFullRegion=*/true, SideEffects::DefaultResource::get());
  }

  auto flags = op.getFlags();
  bool isInit = false;
  for (auto flag : flags) {
    if (cast<BrgemmFlagsAttr>(flag).getValue() == BrgemmFlags::BETA_0) {
      isInit = true;
      break;
    }
  }

  assert(op.getDpsInitsMutable().size() == 1 &&
         "Expecting single DPS init operand");
  OpOperand &operand = op.getDpsInitsMutable()[0];
  if (!llvm::isa<MemRefType>(operand.get().getType()))
    return;
  if (!isInit) {
    effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage=*/0,
                       /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
}

static inline ArrayRef<int64_t> getShapedValueShape(Value val) {
  assert((llvm::isa<TensorType>(val.getType()) ||
          llvm::isa<MemRefType>(val.getType())) &&
         "Expecting shaped value");
  if (auto tensorTy = dyn_cast_or_null<TensorType>(val.getType()))
    return tensorTy.getShape();
  auto memrefTy = dyn_cast_or_null<MemRefType>(val.getType());
  return memrefTy.getShape();
}

LogicalResult BrgemmOp::verify() {
  BrgemmOp op = *this;

  size_t expectedInputSize = 2;
  SmallVector<Value, 2> ins;
  for (auto in : op.getInputs())
    ins.push_back(in);
  Value out = op.getInit();
  ArrayRef<int64_t> batchDims = op.getBatchDims();
  ArrayRef<int64_t> leadingDims = op.getLeadingDims();
  if (ins.size() != expectedInputSize &&
      batchDims.size() != expectedInputSize &&
      leadingDims.size() != expectedInputSize)
    return op.emitOpError()
           << "expect inputs and its related info to be size 2\n";

  auto elemTypeA = getElementTypeOrSelf(ins[0]);
  auto elemTypeB = getElementTypeOrSelf(ins[1]);
  auto elemTypeC = getElementTypeOrSelf(out);
  if (!isTypeSupported(elemTypeC, elemTypeA, elemTypeB))
    return op.emitOpError() << "unsupported input matrix types\n";

  ArrayRef<int64_t> dimA = getShapedValueShape(ins[0]);
  ArrayRef<int64_t> dimB = getShapedValueShape(ins[1]);
  ArrayRef<int64_t> dimC = getShapedValueShape(out);
  if (dimA.size() != 3)
    return op.emitOpError() << "expect input A to be 3D\n";
  if (!elemTypeB.isF32()) {
    if (dimB.size() != 4 ||
        !isInVnniLayout(dyn_cast<ShapedType>(ins[1].getType())))
      return op.emitOpError()
             << "expect a 4d VNNI input B for non-F32 operand: " << ins[1];
  } else {
    if (dimB.size() != 3)
      return op.emitOpError()
             << "expect a 3d input B for F32 operand: " << ins[1];
  }
  if (dimC.size() != 2)
    return op.emitOpError() << "expect input C to be 2D\n";
  for (auto dim : batchDims)
    if (dim >= 2)
      return op.emitOpError() << "batch dim cannot be last dim, as last dim "
                                 "should be contigious\n";
  for (auto dim : leadingDims)
    if (dim >= 2)
      return op.emitOpError() << "leading dim cannot be last dim, as last dim "
                                 "should be contigious\n";

  auto batchA = dimA[batchDims[0]];
  auto batchB = dimB[batchDims[1]];
  auto majorDimA = dimA[leadingDims[0]];
  auto majorDimB = dimB.size() == 3 ? dimB[leadingDims[1]]
                                    : (dimB[leadingDims[1]] * dimB[3]);
  auto minorDimA = dimA[2];
  auto minorDimB = dimB[2];
  auto majorDimC = dimC[0];
  auto minorDimC = dimC[1];
  if (batchA != batchB)
    return op.emitOpError() << "unmatched batch dim of A and B\n";
  if (minorDimA != majorDimB)
    return op.emitOpError() << "unmatched matmul dim of A and B\n";
  if (majorDimA != majorDimC || minorDimB != minorDimC)
    return op.emitOpError() << "unmatched matmul dim of A, B and C\n";

  return verifyBrgemmFlags(op.getFlags(), op, FLAGS_ASM_NAME);
}

bool BrgemmOp::bufferizesToMemoryRead(OpOperand &opOperand,
                                      const AnalysisState &state) {
  Operation *op = *this;
  auto dpsOp = cast<DestinationStyleOpInterface>(op);
  return !dpsOp.isDpsInit(&opOperand);
}

bool BrgemmOp::bufferizesToMemoryWrite(OpOperand &opOperand,
                                       const AnalysisState &state) {
  Operation *op = *this;
  auto dpsOp = cast<DestinationStyleOpInterface>(op);
  return dpsOp.isDpsInit(&opOperand);
}

bool BrgemmOp::bufferizesToElementwiseAccess(const AnalysisState &state,
                                             ArrayRef<OpOperand *> opOperands) {
  // This op contains non-parallel reduction loops,
  // should return `false` per linalg implementation
  return false;
}

AliasingValueList BrgemmOp::getAliasingValues(OpOperand &opOperand,
                                              const AnalysisState &state) {
  // This implementation refers to linalg's usage of
  // ` DstBufferizableOpInterfaceExternalModel`
  Operation *op = *this;
  // Output operands alias with their respective tied OpResults.
  auto dstOp = cast<DestinationStyleOpInterface>(op);
  if (dstOp.isDpsInit(&opOperand))
    return {{dstOp.getTiedOpResult(&opOperand), BufferRelation::Equivalent}};
  return {};
}

LogicalResult BrgemmOp::bufferize(RewriterBase &rewriter,
                                  const BufferizationOptions &options) {
  // This implementation refers to linalg's
  // `bufferizeDestinationStyleOpInterface`
  Operation *op = *this;
  auto dpsOp = cast<DestinationStyleOpInterface>(op);

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(dpsOp);

  // Nothing to do. This op is already bufferized.
  if (dpsOp.hasPureBufferSemantics())
    return success();

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!dpsOp.hasPureTensorSemantics())
    return emitError() << "op does not have pure tensor semantics";

  // New input operands for the cloned op.
  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(dpsOp.getNumDpsInputs());
  for (OpOperand *opOperand : dpsOp.getDpsInputOperands()) {
    FailureOr<Value> buffer = getBuffer(rewriter, opOperand->get(), options);
    if (failed(buffer))
      return failure();
    newInputBuffers.push_back(*buffer);
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : dpsOp->getOpResults()) {
    OpOperand *opOperand = dpsOp.getDpsInitOperand(opResult.getResultNumber());
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, opOperand->get(), options);
    if (failed(resultBuffer))
      return failure();
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Merge input/output operands.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(dpsOp);
  // Clone the op, but use the new operands. Since the new op does not have any
  // tensor results, it does not return anything.
  OperationState state(dpsOp->getLoc(), dpsOp->getName(), newOperands,
                       TypeRange{}, dpsOp->getAttrs());
  Operation *newOp = Operation::create(state);

  // We don't want the rewriter tracks an incomplete operation, so insert new
  // operation after op was fully constructed.
  rewriter.insert(newOp);

  // Replace the results of the old op with the new output buffers.
  replaceOpWithBufferizedValues(rewriter, dpsOp, newOutputBuffers);

  return success();
}

/////////////////////////////////////////////////////
// Start of BrgemmDispatchOp

void BrgemmDispatchOp::print(OpAsmPrinter &printer) {
  BrgemmDispatchOp op = *this;

  printer << " [" << op.getInputs() << ']';

  auto getOpFlags = [this]() -> ArrayAttr { return this->getFlags(); };
  printFlagsImpl<BrgemmFlagsAttr>(printer, getOpFlags, FLAGS_ASM_NAME);

  printDataTypeImpl<BrgemmDispatchOp>(printer, *this);
}

ParseResult BrgemmDispatchOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  auto &builder = parser.getBuilder();
  result.addTypes(builder.getIntegerType(64));

  DenseI64ArrayAttr inputAttr;
  if (parser.parseCustomAttributeWithFallback(
          inputAttr, Type{}, INPUTS_ATTR_NAME, result.attributes))
    return failure();
  if (failed(parseFlagsImpl<BrgemmFlags>(parser, result, FLAGS_ASM_NAME)))
    return failure();
  if (failed(parseArrayAttrImpl<TypeAttr>(parser, result, DATA_TYPE_ASM_NAME,
                                          DATA_TYPE_ASM_NAME)))
    return failure();
  return success();
}

static LogicalResult verifyBrgemmDataTypes(ArrayAttr dtypes,
                                           BrgemmDispatchOp op) {
  if (dtypes.size() != 2)
    return op->emitOpError() << "data types size should be 2";

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
                      return type_pair.first == dtypes[0] &&
                             type_pair.second == dtypes[1];
                    }))
    return op->emitOpError() << "invalid data type pair";

  return success();
}

LogicalResult BrgemmDispatchOp::verify() {
  BrgemmDispatchOp op = *this;
  // 'inputs' = [m, n, k, lda, ldb, ldc, stride_a, stride_b] for BRGEMM.
  size_t expected = 8;
  size_t numInputs = op.getInputs().size();
  if (numInputs != expected)
    return op.emitOpError()
           << "expect " << expected << " args but got: " << numInputs;
  // Verify data types
  if (failed(verifyBrgemmDataTypes(op.getDataType(), op)))
    return failure();

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
  return verifyBrgemmFlags(op.getFlags(), op, FLAGS_ASM_NAME);
}

/////////////////////////////////////////////////////
// Start of BrgemmExecuteOp

LogicalResult BrgemmExecuteOp::verify() {
  BrgemmExecuteOp &brgemmOp = *this;

  SmallVector<Value> inputs = brgemmOp.getInputs();
  // inputs for BRGEMM: kernel id, A memref, B memref, C memref, batch_size,
  // addr_len
  if (inputs.size() != 6)
    return brgemmOp.emitOpError()
           << "expect 6" << " inputs but got " << inputs.size();
  // Verify the dispatch to be an i64.
  Value dispatch = brgemmOp.getDispatch();
  if (!dispatch.getType().isInteger(64))
    return brgemmOp.emitOpError()
           << "expect an i64 but got " << dispatch.getType()
           << " for operand 0 (dispatch)";

  // Verify whether memref types are supported
  SmallVector<Value> memrefOperands = {
      brgemmOp.getOperandA(), brgemmOp.getOperandB(), brgemmOp.getOutput()};
  SmallVector<Type> typeOperands = {
      getElementTypeOrSelf(memrefOperands[0].getType()),
      getElementTypeOrSelf(memrefOperands[1].getType()),
      getElementTypeOrSelf(memrefOperands[2].getType())};
  if (!isTypeSupported(typeOperands[2], typeOperands[0], typeOperands[1]))
    return brgemmOp.emitOpError()
           << "operands types: " << typeOperands[0] << " X " << typeOperands[1]
           << " -> " << typeOperands[2] << " are unsupported";

  // Verify the rank of the shaped operand A.
  auto memrefTypeA = dyn_cast<MemRefType>(memrefOperands[0].getType());
  if (memrefTypeA.getRank() != 3)
    return brgemmOp.emitOpError()
           << "expect a 3d memref for operand A: " << memrefTypeA;

  // Verify the rank of the shaped operand B.
  auto memrefTypeB = dyn_cast<MemRefType>(memrefOperands[1].getType());
  auto dtypeB = typeOperands[1];
  if (!dtypeB.isF32()) {
    if (memrefTypeB.getRank() != 4 || !isInVnniLayout(memrefTypeB))
      return brgemmOp.emitOpError()
             << "expect a 4d VNNI memref for non-F32 operand: " << memrefTypeB;
  } else {
    if (memrefTypeB.getRank() != 3)
      return brgemmOp.emitOpError()
             << "expect a 3d memref for F32 operand: " << memrefTypeB;
  }

  // Verify the rank of the shaped operand C.
  auto memrefTypeC = dyn_cast<MemRefType>(memrefOperands[2].getType());
  if (memrefTypeC.getRank() != 2)
    return brgemmOp.emitOpError()
           << "expect a 2d memref for operand C: " << memrefTypeC;

  // Verify the batch and addrLen to be i64.
  Value batch = brgemmOp.getBatch();
  if (!batch.getType().isInteger(64))
    return brgemmOp.emitOpError() << "expect an i64 but got " << batch.getType()
                                  << " for operand 4 (batch)";
  Value addrLen = brgemmOp.getAddrLen();
  if (!addrLen.getType().isInteger(64))
    return brgemmOp.emitOpError()
           << "expect an i64 but got " << addrLen.getType()
           << " for operand 5 (addrLen)";
  return success();
}

} // namespace microkernel
} // namespace mlir
