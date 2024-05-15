//===- CPURuntimeOps.cpp - CPU Runtime Ops ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/CPURuntime/IR/CPURuntimeOps.h"
#include "gc/Dialect/CPURuntime/IR/CPURuntimeDialect.h"

#define GET_OP_CLASSES
#include "gc/Dialect/CPURuntime/IR/CPURuntimeOps.cpp.inc"

#include <llvm/Support/Debug.h>

namespace mlir {
using namespace bufferization;

namespace cpuruntime {

void AtParallelExitOp::build(OpBuilder &b, OperationState &result) {
  OpBuilder::InsertionGuard g(b);
  Region *bodyRegion = result.addRegion();
  b.createBlock(bodyRegion);
}

void AtParallelExitOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult AtParallelExitOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  auto &builder = parser.getBuilder();

  SmallVector<OpAsmParser::Argument, 8> regionOperands;
  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parser.parseRegion(*region, regionOperands))
    return failure();

  if (region->empty())
    OpBuilder(builder.getContext()).createBlock(region.get());
  result.addRegion(std::move(region));

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

} // namespace cpuruntime
} // namespace mlir