//===-- LinalgxOps.cpp - linalgx dialect ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::linalgx;

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

void AttentionOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TypeRange results, Value query, Value key, Value value,
                        Value scale, Value output, ArrayAttr indexingMaps,
                        std::optional<Value> mask) {
  Value maskIn = mask.value_or(Value());
  build(odsBuilder, odsState, results, query, key, value, scale, maskIn, output,
        indexingMaps, DictionaryAttr());
}

void AttentionOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TypeRange results, ValueRange inputOperands,
                        ValueRange initOperands, ArrayAttr indexingMaps) {
  assert(inputOperands.size() < 6);
  assert(initOperands.size() == 1);
  Value mask = inputOperands.size() > 4 ? inputOperands[4] : Value();
  build(odsBuilder, odsState, results, inputOperands[0], inputOperands[1],
        inputOperands[2], inputOperands[3], mask, initOperands[0], indexingMaps,
        DictionaryAttr());
}

LogicalResult AttentionOp::verify() {
  // TODO: verify the correctness of indexing maps and shapes of operands.
  return success();
}

ArrayRef<int64_t> AttentionOp::getShape(OpOperand *opOperand) {
  assert(opOperand->getOwner() == this->getOperation());
  Type t = opOperand->get().getType();
  // A VectorType is an elemental type, do not consider its rank for the
  // operand.
  if (isa<VectorType>(t)) return {};
  if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
    // Failsafe.
    assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
           "expected a ranked tensor or memref in LinalgInterface::getRank");
    return shapedType.getShape();
  }
  return {};
}

MutableOperandRange AttentionOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/getMask() ? 5 : 4,
                             /*numInits=*/1);
}

SmallVector<AffineMap> AttentionOp::getIndexingMapsArray() {
  return SmallVector<AffineMap>(
      getIndexingMaps().getAsValueRange<AffineMapAttr>());
}

void AttentionOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  // FIXME: add canonicalization patterns
  // patterns.insert<StaticizeLinalgExtOp<AttentionOp>>(ctx);
}

AffineMap AttentionOp::getMatchingIndexingMap(OpOperand *operand) {
  return *(getIndexingMaps().getAsValueRange<AffineMapAttr>().begin() +
           operand->getOperandNumber());
}

//===----------------------------------------------------------------------===//
// GroupedMatmulOp
//===----------------------------------------------------------------------===//

// Keep the single-branch assembly unchanged while representing fused branches
// as (lhs, weights) pairs followed by the one shared offsets operand.
static ParseResult parseGroupedMatmulInputs(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &lhs,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &weights,
    OpAsmParser::UnresolvedOperand &offsets, SmallVectorImpl<Type> &lhsTypes,
    SmallVectorImpl<Type> &weightsTypes, Type &offsetsType) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> types;
  if (parser.parseOperandList(operands) || parser.parseColonTypeList(types))
    return failure();
  if (operands.size() < 3 || operands.size() % 2 != 1)
    return parser.emitError(
        parser.getCurrentLocation(),
        "expected (lhs, weights) pairs followed by offsets");
  if (types.size() != operands.size())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected one type per input operand");
  for (unsigned i = 0; i + 1 < operands.size(); i += 2) {
    lhs.push_back(operands[i]);
    weights.push_back(operands[i + 1]);
    lhsTypes.push_back(types[i]);
    weightsTypes.push_back(types[i + 1]);
  }
  offsets = operands.back();
  offsetsType = types.back();
  return success();
}

static void printGroupedMatmulInputs(OpAsmPrinter &printer, Operation *,
                                     OperandRange lhs, OperandRange weights,
                                     Value offsets, TypeRange lhsTypes,
                                     TypeRange weightsTypes, Type offsetsType) {
  for (auto [a, w] : llvm::zip_equal(lhs, weights))
    printer << a << ", " << w << ", ";
  printer << offsets << " : ";
  for (auto [a, w] : llvm::zip_equal(lhsTypes, weightsTypes))
    printer << a << ", " << w << ", ";
  printer << offsetsType;
}

static ParseResult parseGroupedMatmulResults(OpAsmParser &parser,
                                             SmallVectorImpl<Type> &types) {
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseTypeList(types) || parser.parseRParen()) return failure();
    return success();
  }
  return parser.parseTypeList(types);
}

static void printGroupedMatmulResults(OpAsmPrinter &printer, Operation *,
                                      TypeRange types) {
  if (types.size() > 1) printer << '(';
  printer << types;
  if (types.size() > 1) printer << ')';
}

LogicalResult GroupedMatmulOp::verify() {
  unsigned branches = getNumBranches();
  if (!branches || getLhs().size() != branches ||
      getWeights().size() != branches)
    return emitOpError(
        "expects equally sized, non-empty lhs, weights and outputs lists");
  auto offTy = dyn_cast<ShapedType>(getOffsets().getType());
  if (!offTy || !offTy.hasRank() || offTy.getRank() != 1)
    return emitOpError("expects rank-1 offsets [E]");
  if (!offTy.getElementType().isSignlessInteger())
    return emitOpError("expects integer offsets");
  if (getIndexingMaps().size() != 3 * branches + 1)
    return emitOpError("expects one indexing map per shaped operand; "
                       "'max_group_size' is a scalar and has none");

  bool sharedLhs = llvm::all_of(
      getLhs(), [&](Value value) { return value == getLhs().front(); });
  bool sharedWeights = llvm::all_of(
      getWeights(), [&](Value value) { return value == getWeights().front(); });
  if (!sharedLhs && !sharedWeights)
    return emitOpError("fused branches must share their lhs or weights value");

  // Static shape agreement. Dynamic dims are unchecked.
  auto sameIfStatic = [](int64_t a, int64_t b) {
    return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
  };
  SmallVector<ShapedType> lhsTypes, weightsTypes, outputTypes;
  for (unsigned i = 0; i < branches; ++i) {
    auto lhsTy = dyn_cast<ShapedType>(getLhs()[i].getType());
    auto weiTy = dyn_cast<ShapedType>(getWeights()[i].getType());
    auto outTy = dyn_cast<ShapedType>(getOutputs()[i].getType());
    if (!lhsTy || !lhsTy.hasRank() || lhsTy.getRank() != 2)
      return emitOpError("expects rank-2 lhs [total, K]");
    if (!weiTy || !weiTy.hasRank() || weiTy.getRank() != 3)
      return emitOpError("expects rank-3 weights [E, N, K]");
    if (!outTy || !outTy.hasRank() || outTy.getRank() != 2)
      return emitOpError("expects rank-2 output [total, N]");
    if (!sameIfStatic(lhsTy.getDimSize(0), outTy.getDimSize(0)))
      return emitOpError(
          "expects lhs and output to have the same number of rows");
    if (!sameIfStatic(lhsTy.getDimSize(1), weiTy.getDimSize(2)))
      return emitOpError("expects lhs' K to match weights' innermost dim");
    if (!sameIfStatic(outTy.getDimSize(1), weiTy.getDimSize(1)))
      return emitOpError("expects output's N to match weights' middle dim");
    if (!sameIfStatic(offTy.getDimSize(0), weiTy.getDimSize(0)))
      return emitOpError("expects one offset per expert");
    lhsTypes.push_back(lhsTy);
    weightsTypes.push_back(weiTy);
    outputTypes.push_back(outTy);
    if (i == 0) continue;
    for (auto [type, first] : {std::pair{lhsTy, lhsTypes.front()},
                               std::pair{weiTy, weightsTypes.front()},
                               std::pair{outTy, outputTypes.front()}}) {
      if (type.getElementType() != first.getElementType())
        return emitOpError("fused branches must have matching element types");
      for (auto [dim, firstDim] :
           llvm::zip_equal(type.getShape(), first.getShape()))
        if (!sameIfStatic(dim, firstDim))
          return emitOpError(
              "fused branches must have a common iteration domain");
    }
  }

  if (branches > 1 && (!isa<FloatType>(lhsTypes.front().getElementType()) ||
                       !isa<FloatType>(weightsTypes.front().getElementType()) ||
                       !isa<FloatType>(outputTypes.front().getElementType())))
    return emitOpError("fused grouped matmul only supports "
                       "floating-point inputs and outputs");

  AffineExpr e, m, n, k;
  bindDims(getContext(), e, m, n, k);
  auto map = [&](ArrayRef<AffineExpr> exprs) {
    return AffineMap::get(4, 0, exprs, getContext());
  };
  for (unsigned i = 0; i < branches; ++i)
    if (getLhsMap(i) != map({m, k}) || getWeightsMap(i) != map({e, n, k}) ||
        getOutputMap(i) != map({m, n}))
      return emitOpError("expects canonical grouped matmul indexing maps");
  if (getOffsetsMap() != map({e}))
    return emitOpError("expects offsets indexing map (e, m, n, k) -> (e)");

  bool tensorSemantics = isa<RankedTensorType>(getOutputs().front().getType());
  for (Value operand : getOperands()) {
    if (operand == getMaxGroupSize()) continue;
    if (tensorSemantics ? !isa<RankedTensorType>(operand.getType())
                        : !isa<MemRefType>(operand.getType()))
      return emitOpError("expects uniformly tensor or memref shaped operands");
  }
  if (getNumResults() != (tensorSemantics ? branches : 0))
    return emitOpError(
        "expects one result per tensor output and no memref results");
  for (auto [result, output] : llvm::zip(getResults(), getOutputs()))
    if (result.getType() != output.getType())
      return emitOpError("expects result types to match output types");
  return success();
}

ArrayRef<int64_t> GroupedMatmulOp::getShape(OpOperand *opOperand) {
  assert(opOperand->getOwner() == this->getOperation());
  if (auto shapedType = dyn_cast<ShapedType>(opOperand->get().getType()))
    return shapedType.getShape();
  return {};
}

MutableOperandRange GroupedMatmulOp::getDpsInitsMutable() {
  return getOutputsMutable();
}

void GroupedMatmulOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (OpOperand *operand : getDpsInputOperands()) {
    if (isa<MemRefType>(operand->get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), operand);
  }
  for (OpOperand &output : getDpsInitsMutable()) {
    if (!isa<MemRefType>(output.get().getType())) continue;
    // Outputs are read as initial accumulators, then written with the results.
    effects.emplace_back(MemoryEffects::Read::get(), &output);
    effects.emplace_back(MemoryEffects::Write::get(), &output);
  }
}

Speculation::Speculatability GroupedMatmulOp::getSpeculatability() {
  return hasPureTensorSemantics() ? Speculation::Speculatable
                                  : Speculation::NotSpeculatable;
}

SmallVector<AffineMap> GroupedMatmulOp::getIndexingMapsArray() {
  return SmallVector<AffineMap>(
      getIndexingMaps().getAsValueRange<AffineMapAttr>());
}

AffineMap GroupedMatmulOp::getMatchingIndexingMap(OpOperand *operand) {
  assert(operand->getOwner() == this->getOperation());
  // The optional scalar bound has no indexing map.
  unsigned pos = operand->getOperandNumber();
  unsigned shapedInputs = getLhs().size() + getWeights().size() + 1;
  if (getMaxGroupSize()) {
    if (pos == shapedInputs)
      return AffineMap::get(getIterationDomainRank(), 0, getContext());
    if (pos > shapedInputs) --pos;
  }
  return getIndexingMapsArray()[pos];
}

/////// Operations corresponding to library calls defined with Tablegen ////////

#define GET_OP_CLASSES
#include "gc/Dialect/Linalgx/LinalgxOps.cpp.inc"
