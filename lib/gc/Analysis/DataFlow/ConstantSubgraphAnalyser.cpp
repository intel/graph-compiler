//===-- ConstantSubgraphAnalyser.cpp - Constant subgraph  -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <unordered_set>

#include "gc/Analysis/DataFlow/ConstantSubgraphAnalyser.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "in-constant-subgraph"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// IsConstantTensor
//===----------------------------------------------------------------------===//

void IsConstantTensor::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "<UNINITIALIZED>";
    return;
  }
  os << getIsConstantTensor();
}

//===----------------------------------------------------------------------===//
// ConstantSubgraphAnalyser
//===----------------------------------------------------------------------===//

void ConstantSubgraphAnalyser::visitOperation(
    Operation *op, ArrayRef<const Lattice<IsConstantTensor> *> operands,
    ArrayRef<Lattice<IsConstantTensor> *> results) {
  LLVM_DEBUG(llvm::dbgs() << "ConstantSubgraphAnalyser: Visiting operation:\n"
                          << *op << "\n");

  bool in = true;
  if (op->hasTrait<OpTrait::ConstantLike>()) {
    LLVM_DEBUG(llvm::dbgs() << "Curr op is a Constant op\n");
    in = true;
  } else if (operands.empty()) { // For example, tensor.empty()
    LLVM_DEBUG(llvm::dbgs() << "Curr op has 0 operand, constant\n");
    in = true;
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Curr op has " << operands.size()
                            << " operands, check if constant\n");
    for (auto *operandLattice : operands) {
      auto operandState = operandLattice->getValue().getIsConstantTensor();
      LLVM_DEBUG(llvm::dbgs() << "Operand: " << operandLattice->getPoint()
                              << ", lattice value: " << operandState << "\n");
      if (!operandState) {
        in = false;
        break;
      }
    }
  }

  // lattice in results should be in unintialized state.
  if (!in) {
    LLVM_DEBUG(llvm::dbgs() << "Curr op not in constant subgraph\n");
    for (auto lattice : results) {
      propagateIfChanged(lattice, lattice->join(IsConstantTensor(true, false)));
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Curr op in constant subgraph\n");
    for (auto lattice : results) {
      propagateIfChanged(lattice, lattice->join(IsConstantTensor(true, true)));
    }
  }
}

void ConstantSubgraphAnalyser::setToEntryState(
    Lattice<IsConstantTensor> *lattice) {
  if (auto blockArg = cast<BlockArgument>(lattice->getPoint())) {
    auto parentOp = blockArg.getParentBlock()->getParentOp();
    auto parentOpAttr = parentOp->getAttrDictionary();

    std::unordered_set<int> constArgsIndexes;
    std::optional<NamedAttribute> compiletimeConstArgs =
        parentOpAttr.getNamed("compiletime_const_args_index");
    if (compiletimeConstArgs.has_value()) {
      for (auto id :
           llvm::dyn_cast<ArrayAttr>(compiletimeConstArgs->getValue())) {
        constArgsIndexes.insert(llvm::cast<IntegerAttr>(id).getInt());
      }
    }
    std::optional<NamedAttribute> runtimeConstArgs =
        parentOpAttr.getNamed("runtime_const_args_index");
    if (runtimeConstArgs.has_value()) {
      for (auto id : llvm::dyn_cast<ArrayAttr>(runtimeConstArgs->getValue())) {
        constArgsIndexes.insert(llvm::cast<IntegerAttr>(id).getInt());
      }
    }

    if (constArgsIndexes.count(blockArg.getArgNumber())) {
      LLVM_DEBUG(llvm::dbgs() << "Block argument: " << blockArg
                              << " is marked as constant\n");
      propagateIfChanged(lattice, lattice->join(IsConstantTensor(true, true)));
      return;
    }
    propagateIfChanged(lattice, lattice->join(IsConstantTensor(true, false)));
  } else {
    propagateIfChanged(lattice,
                       lattice->join(IsConstantTensor::getUninitialized()));
  }
}

//===----------------------------------------------------------------------===//
// RunConstantSubgraphAnalyser
//===----------------------------------------------------------------------===//

/// Get the operations whose inputs and outputs are all constant values.
/// These operations will be put into a seperate subgraph.
void RunConstantSubgraphAnalyser::getConstantSubgraph(DataFlowSolver &solver,
                                                      Operation *topFunc) {
  OpBuilder builder(topFunc->getContext());
  SmallVector<Operation *> constantOperations;

  Block &block = topFunc->getRegions().front().getBlocks().front();
  for (Operation &op : llvm::make_early_inc_range(block)) {
    // If all the result values of a op are const, we mark this op as const.
    bool resultsAllConstant = true;
    if (op.getNumResults() == 0) {
      continue;
    }
    for (Value res : op.getResults()) {
      auto *lattice = solver.lookupState<Lattice<IsConstantTensor>>(res);
      if (!lattice || lattice->getValue().isUninitialized()) {
        resultsAllConstant = false;
        break;
      }
      const IsConstantTensor &latticeValue = lattice->getValue();
      if (!latticeValue.getIsConstantTensor()) {
        resultsAllConstant = false;
        break;
      }
    }
    if (resultsAllConstant) {
      op.setAttr("onednn_graph.in_const_subgraph", builder.getBoolAttr(true));
      constantOperations.push_back(&op);
    }
  }

  if (constantOperations.empty()) {
    return;
  }
}

RunConstantSubgraphAnalyser::RunConstantSubgraphAnalyser() {
  solver.load<DeadCodeAnalysis>();
  solver.load<ConstantSubgraphAnalyser>();
}

void RunConstantSubgraphAnalyser::run(Operation *op) {
  if (failed(solver.initializeAndRun(op))) {
    return;
  }
  getConstantSubgraph(solver, op);
}

bool RunConstantSubgraphAnalyser::getIsConstantTensor(Value val) {
  auto *lattice = solver.lookupState<Lattice<IsConstantTensor>>(val);
  const IsConstantTensor &latticeValue = lattice->getValue();
  return latticeValue.getIsConstantTensor();
}