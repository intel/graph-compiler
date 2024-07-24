//===- ConstantSubgraphAnalyser.cpp - Constant subgraph analysis ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
#include <cassert>

#define DEBUG_TYPE "in-constant-subgraph"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// InConstantSubgraph
//===----------------------------------------------------------------------===//

void InConstantSubgraph::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "<UNINITIALIZED>";
    return;
  }
  os << getInConstantSubgraph();
  return;
}

//===----------------------------------------------------------------------===//
// ConstantSubgraphAnalyser
//===----------------------------------------------------------------------===//

void ConstantSubgraphAnalyser::visitOperation(
    Operation *op, ArrayRef<const Lattice<InConstantSubgraph> *> operands,
    ArrayRef<Lattice<InConstantSubgraph> *> results) {
  LLVM_DEBUG(llvm::dbgs() << "ConstantSubgraphAnalyser: Visiting operation:\n"
                          << *op << "\n");

  bool in = true;
  if (op->hasTrait<OpTrait::ConstantLike>()) {
    LLVM_DEBUG(llvm::dbgs() << "Curr op is a Constant op\n");
    in = true;
  } else if (operands.size() == 0) { // For example, tensor.empty()
    LLVM_DEBUG(llvm::dbgs() << "Curr op has 0 operand, constant\n");
    in = true;
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Curr op has " << operands.size()
                            << " operands, check if constant\n");
    for (auto *operandLattice : operands) {
      auto operandState = operandLattice->getValue().getInConstantSubgraph();
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
      propagateIfChanged(lattice,
                         lattice->join(InConstantSubgraph(true, false)));
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Curr op in constant subgraph\n");
    for (auto lattice : results) {
      propagateIfChanged(lattice,
                         lattice->join(InConstantSubgraph(true, true)));
    }
  }
}

void ConstantSubgraphAnalyser::setToEntryState(
    Lattice<InConstantSubgraph> *lattice) {
  if (auto blockArg = cast<BlockArgument>(lattice->getPoint())) {
    auto parent_op = blockArg.getParentBlock()->getParentOp();
    auto parent_op_attr = parent_op->getAttrDictionary();
    std::optional<NamedAttribute> const_args =
        parent_op_attr.getNamed("onednn_graph.const_args");
    if (const_args.has_value()) {
      ArrayAttr const_args_indexes =
          llvm::dyn_cast<ArrayAttr>(const_args->getValue());
      for (auto id : const_args_indexes) {
        auto idint = llvm::cast<IntegerAttr>(id).getInt();
        if (blockArg.getArgNumber() == idint) {
          LLVM_DEBUG(llvm::dbgs() << "Block argument: " << blockArg
                                  << " is marked as constant\n");
          propagateIfChanged(lattice,
                             lattice->join(InConstantSubgraph(true, true)));
          return;
        }
      }
    }
    propagateIfChanged(lattice, lattice->join(InConstantSubgraph(true, false)));
  } else {
    propagateIfChanged(lattice,
                       lattice->join(InConstantSubgraph::getUninitialized()));
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
      auto *lattice = solver.lookupState<Lattice<InConstantSubgraph>>(res);
      if (!lattice || lattice->getValue().isUninitialized()) {
        resultsAllConstant = false;
        break;
      }
      const InConstantSubgraph &latticeValue = lattice->getValue();
      if (!latticeValue.getInConstantSubgraph()) {
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

void RunConstantSubgraphAnalyser::run(Operation *topFunc) {
  if (failed(solver.initializeAndRun(topFunc))) {
    return;
  }
  getConstantSubgraph(solver, topFunc);
}

bool RunConstantSubgraphAnalyser::getInConstantSubgraph(Value val) {
  auto *lattice = solver.lookupState<Lattice<InConstantSubgraph>>(val);
  const InConstantSubgraph &latticeValue = lattice->getValue();
  return latticeValue.getInConstantSubgraph();
}