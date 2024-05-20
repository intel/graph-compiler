//===- LinalgxOps.h - linalgx dialect ops -----------------------*- C++ -*-===//
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

/////// Operations corresponding to library calls defined with Tablegen ////////

#define GET_OP_CLASSES
#include "gc/Dialect/Linalgx/LinalgxOps.cpp.inc"

#define GET_OP_CLASSES
#include "gc/Dialect/Linalgx/LinalgxStructuredOps.cpp.inc"
