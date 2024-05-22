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

//===----------------------------------------------------------------------===//
// Builder helper from mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp
//===----------------------------------------------------------------------===//

#include "LinalgOps.cpp.inc"

using namespace mlir;
using namespace mlir::linalgx;

//===----------------------------------------------------------------------===//
// SigmoidOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> SigmoidOp::getIteratorTypesArray() {
  int64_t rank = getRank(getDpsInitOperand(0));
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

ArrayAttr SigmoidOp::getIndexingMaps() {
  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  AffineMap tensorMap =
      AffineMap::getMultiDimIdentityMap(getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand &opOperand : getOperation()->getOpOperands())
    indexingMaps.push_back(getRank(&opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

void SigmoidOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                              ArrayRef<NamedAttribute> attrs) {
  assert(2 > 0 && block.getNumArguments() == 2 &&
         "ExpOp regionBuilder expects 2 (>=0) args");
  RegionBuilderHelper helper(b, block);
  SmallVector<Value> yields;

  auto elemTy = block.getArgument(1).getType();
  Value value1 = helper.buildUnaryFn(UnaryFn::negf, block.getArgument(0));
  Value value2 = helper.buildUnaryFn(UnaryFn::exp, value1);
  Value value3 =
      b.create<arith::ConstantOp>(b.getUnknownLoc(), b.getFloatAttr(elemTy, 1));
  Value value4 = helper.buildBinaryFn(BinaryFn::add, value3, value2);
  Value value5 =
      b.create<arith::ConstantOp>(b.getUnknownLoc(), b.getFloatAttr(elemTy, 1));
  Value value6 = helper.buildBinaryFn(BinaryFn::div, value5, value4);

  yields.push_back(value6);
  helper.yieldOutputs(yields);
}

ParseResult SigmoidOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result, SigmoidOp::getNumRegionArgs(),
                                  SigmoidOp::getRegionBuilder());
}

void SigmoidOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult SigmoidOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void SigmoidOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (hasPureTensorSemantics())
    return;
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

//===----------------------------------------------------------------------===//
// Mm2DVnniOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Mm2DVnniOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::parallel,  utils::IteratorType::reduction,
      utils::IteratorType::reduction, utils::IteratorType::reduction};
}

static SmallVector<AffineExpr> getSymbolBindings(Mm2DVnniOp self) {
  MLIRContext *context = self.getContext();

  auto vnniShape = ShapeAdaptor(self.getInputs()[1].getType());

  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

  int64_t cst2 = vnniShape.getDimSize(3);
  exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

  int64_t cst4 = vnniShape.getDimSize(2);
  exprs.push_back(getAffineConstantExpr(cst4, context));

  int64_t cst5 = vnniShape.getDimSize(4);
  exprs.push_back(getAffineConstantExpr(cst5, context));
  return exprs;
}

ArrayAttr Mm2DVnniOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  static auto mapA = "affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, "
                     "s4, s5] -> (d0, (d3 * s4 + d4) * s5 + d5)>";
  static auto mapB = "affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, "
                     "s4, s5] -> (d1, d3, d4, d2, d5)>";
  static auto mapC = "affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, "
                     "s4, s5] -> (d0, d1 * s2 + d2)>";
  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(llvm::cast<AffineMapAttr>(mlir::parseAttribute(mapA, context))
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(llvm::cast<AffineMapAttr>(mlir::parseAttribute(mapB, context))
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(llvm::cast<AffineMapAttr>(mlir::parseAttribute(mapC, context))
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

void Mm2DVnniOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                               ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Mm2DVnniOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(b, block);
  SmallVector<Value> yields;

  Value value1 =
      helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(),
                         block.getArgument(0));
  Value value2 =
      helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(),
                         block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 =
      helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Mm2DVnniOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
                                  Mm2DVnniOp::getNumRegionArgs(),
                                  Mm2DVnniOp::getRegionBuilder());
}

void Mm2DVnniOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Mm2DVnniOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void Mm2DVnniOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (hasPureTensorSemantics())
    return;
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

LogicalResult Mm2DVnniOp::verify() {
  // A[M, K]
  // B[N0, K0, K1, N1, K2]
  // C[M, N]
  auto shapeA = ShapeAdaptor(getInputs()[0].getType());
  auto shapeB = ShapeAdaptor(getInputs()[1].getType());
  auto shapeC = ShapeAdaptor(getOutputs()[0].getType());
  // check rank
  auto hasRank = shapeA.hasRank() && shapeB.hasRank() && shapeC.hasRank();
  if (!hasRank)
    return emitOpError() << "input/output must have rank.";
  auto checkRank = (shapeA.getRank() == 2) && (shapeB.getRank() == 5) &&
                   (shapeC.getRank() == 2);
  if (!checkRank)
    return emitOpError() << "not supported input/output shape.";
  // match M, N, K dims
  bool matchM = shapeA.getDimSize(0) == shapeC.getDimSize(0);
  bool matchN =
      (shapeB.getDimSize(0) * shapeB.getDimSize(3)) == shapeC.getDimSize(1);
  bool matchK =
      shapeA.getDimSize(1) ==
      (shapeB.getDimSize(1) * shapeB.getDimSize(2) * shapeB.getDimSize(4));
  bool matchVnni = (shapeB.getDimSize(4) == 2) || (shapeB.getDimSize(4) == 4);
  bool result = matchM && matchN && matchK && matchVnni;
  if (!result)
    return emitOpError() << "input/output dims packing not match.";
  return success();
}

//===----------------------------------------------------------------------===//
// Mm4DVnniOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Mm4DVnniOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::reduction, utils::IteratorType::reduction,
      utils::IteratorType::reduction};
}

static SmallVector<AffineExpr> getSymbolBindings(Mm4DVnniOp self) {
  MLIRContext *context = self.getContext();

  auto vnniShape = ShapeAdaptor(self.getInputs()[1].getType());

  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  exprs.push_back(getAffineSymbolExpr(4, context));
  exprs.push_back(getAffineSymbolExpr(5, context));

  int64_t cst6 = vnniShape.getDimSize(4);
  exprs.push_back(getAffineConstantExpr(cst6, context));
  return exprs;
}

ArrayAttr Mm4DVnniOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  static auto mapA = "affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, "
                     "s4, s5, s6] -> (d0, d4, d2, d5 * s6 + d6)>";
  static auto mapB = "affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, "
                     "s4, s5, s6] -> (d1, d4, d5, d3, d6)>";
  static auto mapC = "affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, "
                     "s4, s5, s6] -> (d0, d1, d2, d3)>";
  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(llvm::cast<AffineMapAttr>(mlir::parseAttribute(mapA, context))
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(llvm::cast<AffineMapAttr>(mlir::parseAttribute(mapB, context))
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(llvm::cast<AffineMapAttr>(mlir::parseAttribute(mapC, context))
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

void Mm4DVnniOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                               ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Mm4DVnniOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(b, block);
  SmallVector<Value> yields;

  Value value1 =
      helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(),
                         block.getArgument(0));
  Value value2 =
      helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(),
                         block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 =
      helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Mm4DVnniOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
                                  Mm4DVnniOp::getNumRegionArgs(),
                                  Mm4DVnniOp::getRegionBuilder());
}

void Mm4DVnniOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Mm4DVnniOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void Mm4DVnniOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (hasPureTensorSemantics())
    return;
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

LogicalResult Mm4DVnniOp::verify() {
  // A[M0, K0, M1, K]
  // B[N0, K0, K1, N1, K2]
  // C[M0, N0, M1, N1]
  auto shapeA = ShapeAdaptor(getInputs()[0].getType());
  auto shapeB = ShapeAdaptor(getInputs()[1].getType());
  auto shapeC = ShapeAdaptor(getOutputs()[0].getType());
  // check rank
  auto hasRank = shapeA.hasRank() && shapeB.hasRank() && shapeC.hasRank();
  if (!hasRank)
    return emitOpError() << "input/output must have rank.";
  auto checkRank = (shapeA.getRank() == 4) && (shapeB.getRank() == 5) &&
                   (shapeC.getRank() == 4);
  if (!checkRank)
    return emitOpError() << "not supported input/output shape.";
  // match M0, M1, N0, N1, K0, K dims
  bool matchM0 = shapeA.getDimSize(0) == shapeC.getDimSize(0);
  bool matchM1 = shapeA.getDimSize(2) == shapeC.getDimSize(2);
  bool matchN0 = shapeB.getDimSize(0) == shapeC.getDimSize(1);
  bool matchN1 = shapeB.getDimSize(3) == shapeC.getDimSize(3);
  bool matchK0 = shapeA.getDimSize(1) == shapeB.getDimSize(1);
  bool matchK =
      shapeA.getDimSize(3) == (shapeB.getDimSize(2) * shapeB.getDimSize(4));
  bool matchVnni = (shapeB.getDimSize(4) == 2) || (shapeB.getDimSize(4) == 4);
  bool result = matchM0 && matchM1 && matchN0 && matchN1 && matchK0 && matchK &&
                matchVnni;
  if (!result)
    return emitOpError() << "input/output dims packing not match.";
  return success();
}

//===----------------------------------------------------------------------===//
// BatchReduceMatmulVnniOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType>
BatchReduceMatmulVnniOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction, utils::IteratorType::reduction,
      utils::IteratorType::reduction};
}

static SmallVector<AffineExpr> getSymbolBindings(BatchReduceMatmulVnniOp self) {
  MLIRContext *context = self.getContext();

  auto vnniShape = ShapeAdaptor(self.getInputs()[1].getType());

  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));

  int64_t cst4 = vnniShape.getDimSize(3);
  exprs.push_back(getAffineConstantExpr(cst4, context));
  return exprs;
}

ArrayAttr BatchReduceMatmulVnniOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  static auto mapA = "affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4] -> "
                     "(d2, d0, d3 * s4 + d4)>";
  static auto mapB = "affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4] -> "
                     "(d2, d3, d1, d4)>";
  static auto mapC = "affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4] -> "
                     "(d0, d1)>";
  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(llvm::cast<AffineMapAttr>(mlir::parseAttribute(mapA, context))
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(llvm::cast<AffineMapAttr>(mlir::parseAttribute(mapB, context))
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(llvm::cast<AffineMapAttr>(mlir::parseAttribute(mapC, context))
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

void BatchReduceMatmulVnniOp::regionBuilder(ImplicitLocOpBuilder &b,
                                            Block &block,
                                            ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "BatchReduceMatmulVnniOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(b, block);
  SmallVector<Value> yields;

  Value value1 =
      helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(),
                         block.getArgument(0));
  Value value2 =
      helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(),
                         block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 =
      helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult BatchReduceMatmulVnniOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
                                  BatchReduceMatmulVnniOp::getNumRegionArgs(),
                                  BatchReduceMatmulVnniOp::getRegionBuilder());
}

void BatchReduceMatmulVnniOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult BatchReduceMatmulVnniOp::fold(FoldAdaptor,
                                            SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void BatchReduceMatmulVnniOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (hasPureTensorSemantics())
    return;
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

LogicalResult BatchReduceMatmulVnniOp::verify() {
  // A[B, M, K]
  // B[B, K0, N, K1]
  // C[M, N]
  auto shapeA = ShapeAdaptor(getInputs()[0].getType());
  auto shapeB = ShapeAdaptor(getInputs()[1].getType());
  auto shapeC = ShapeAdaptor(getOutputs()[0].getType());
  // check rank
  auto hasRank = shapeA.hasRank() && shapeB.hasRank() && shapeC.hasRank();
  if (!hasRank)
    return emitOpError() << "input/output must have rank.";
  auto checkRank = (shapeA.getRank() == 3) && (shapeB.getRank() == 4) &&
                   (shapeC.getRank() == 2);
  if (!checkRank)
    return emitOpError() << "not supported input/output shape.";
  // match B, M, N, K dims
  bool matchB = shapeA.getDimSize(0) == shapeB.getDimSize(0);
  bool matchM = shapeA.getDimSize(1) == shapeC.getDimSize(0);
  bool matchN = shapeB.getDimSize(2) == shapeC.getDimSize(1);
  bool matchK =
      shapeA.getDimSize(2) == (shapeB.getDimSize(1) * shapeB.getDimSize(3));
  bool matchVnni = (shapeB.getDimSize(3) == 2) || (shapeB.getDimSize(3) == 4);
  bool result = matchB && matchM && matchN && matchK && matchVnni;
  if (!result)
    return emitOpError() << "input/output dims packing not match.";
  return success();
}

//===----------------------------------------------------------------------===//
// MultiBatchMatmulOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> MultiBatchMatmulOp::getIteratorTypesArray() {
  int64_t rank = getRank(getDpsInitOperand(0));
  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);
  iteratorTypes.push_back(utils::IteratorType::reduction);
  return iteratorTypes;
}

static SmallVector<AffineExpr> getSymbolBindings(MultiBatchMatmulOp self) {
  MLIRContext *context = self.getContext();
  int64_t symbols = self.getRank(self.getDpsInitOperand(0)) + 1;
  SmallVector<AffineExpr> exprs;
  for (auto dim : llvm::seq<int64_t>(0, symbols)) {
    exprs.push_back(getAffineSymbolExpr(dim, context));
  }
  return exprs;
}

ArrayAttr MultiBatchMatmulOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;
  int64_t symbols = getRank(getDpsInitOperand(0)) + 1;
  int64_t batches = getRank(getDpsInitOperand(0)) - 2;
  MLIRContext *context = getContext();
  // Get affine_map with specified mat dims
  auto getBatchMMAffineMap = [&](int64_t mat1, int64_t mat2) {
    SmallVector<AffineExpr> exprs;
    // batch dims
    for (auto dim : llvm::seq<int64_t>(0, batches)) {
      auto expr = getAffineDimExpr(dim, context);
      exprs.push_back(expr);
    }
    // mat dims
    exprs.push_back(getAffineDimExpr(mat1, context));
    exprs.push_back(getAffineDimExpr(mat2, context));
    return AffineMap::get(symbols, symbols, exprs, context);
  };
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(getBatchMMAffineMap(batches + 0, batches + 2));
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, symbols, 0));
  maps.push_back(getBatchMMAffineMap(batches + 2, batches + 1));
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, symbols, 0));
  maps.push_back(getBatchMMAffineMap(batches + 0, batches + 1));
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, symbols, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

void MultiBatchMatmulOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                                       ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "MultiBatchMatmulOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(b, block);
  SmallVector<Value> yields;

  Value value1 =
      helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(),
                         block.getArgument(0));
  Value value2 =
      helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(),
                         block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 =
      helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult MultiBatchMatmulOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
                                  MultiBatchMatmulOp::getNumRegionArgs(),
                                  MultiBatchMatmulOp::getRegionBuilder());
}

void MultiBatchMatmulOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult MultiBatchMatmulOp::fold(FoldAdaptor,
                                       SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void MultiBatchMatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (hasPureTensorSemantics())
    return;
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

/////// Operations corresponding to library calls defined with Tablegen ////////

#define GET_OP_CLASSES
#include "gc/Dialect/Linalgx/LinalgxOps.cpp.inc"

#define GET_OP_CLASSES
#include "gc/Dialect/Linalgx/LinalgxStructuredOps.cpp.inc"
