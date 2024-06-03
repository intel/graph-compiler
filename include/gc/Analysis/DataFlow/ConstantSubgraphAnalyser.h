//===- ConstantSubgraphAnalyser.h - Constant subgraph analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements constant subgraph analysis. In this file are:
// 1. the lattice value class that represents operations with constant inputs
// and outputs in the program, and
// 2. a sparse constant subgraph analysis.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_CONSTANTSUBGRAPHANALYSER_H
#define MLIR_ANALYSIS_DATAFLOW_CONSTANTSUBGRAPHANALYSER_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include <optional>

namespace mlir {
namespace dataflow {

//===----------------------------------------------------------------------===//
// InConstantSubgraph
//===----------------------------------------------------------------------===//

/// This lattice represents a boolean integer indicating if an operation is with
/// constant inputs and constant outputs and hence in constant subgraph.
class InConstantSubgraph {
public:
  /// Construct as uninitialized.
  explicit InConstantSubgraph() = default;

  /// Construct with a known state.
  explicit InConstantSubgraph(bool initialized, bool inConstantSubgraph)
      : initialized(initialized), inConstantSubgraph(inConstantSubgraph) {}

  /// Get the state. Returns null if no value was determined.
  bool getInConstantSubgraph() const {
    assert(!isUninitialized());
    return inConstantSubgraph;
  }

  /// Compare.
  bool operator==(const InConstantSubgraph &rhs) const {
    return initialized == rhs.initialized &&
           inConstantSubgraph == rhs.inConstantSubgraph;
  }

  void print(raw_ostream &os) const;

  /// Get uninitialized state. This happens when the
  /// state hasn't been set during the analysis.
  static InConstantSubgraph getUninitialized() { return InConstantSubgraph{}; }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !initialized; }

  /// Get unknown state.
  static InConstantSubgraph getUnknown() {
    return InConstantSubgraph{/*initialized=*/false,
                              /*inConstantSubgraph=*/false};
  }

  // Join two states.
  static InConstantSubgraph join(const InConstantSubgraph &lhs,
                                 const InConstantSubgraph &rhs) {
    // if one is uninitialized, use another
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;

    // both are initialized, intersect them
    if (!lhs.isUninitialized() && !rhs.isUninitialized()) {
      return InConstantSubgraph(true, lhs.getInConstantSubgraph() &&
                                          rhs.getInConstantSubgraph());
    }
    return getUninitialized();
  }

private:
  bool initialized = false;
  bool inConstantSubgraph = false;
};

//===----------------------------------------------------------------------===//
// ConstantSubgraphAnalyser
//===----------------------------------------------------------------------===//

class ConstantSubgraphAnalyser
    : public SparseForwardDataFlowAnalysis<Lattice<InConstantSubgraph>> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void visitOperation(Operation *op,
                      ArrayRef<const Lattice<InConstantSubgraph> *> operands,
                      ArrayRef<Lattice<InConstantSubgraph> *> results) override;

  void setToEntryState(Lattice<InConstantSubgraph> *lattice) override;
};

//===----------------------------------------------------------------------===//
// RunConstantSubgraphAnalyser
//===----------------------------------------------------------------------===//

/// Runs constant subgraph analysis on the IR defined by `op`.
struct RunConstantSubgraphAnalyser {
public:
  RunConstantSubgraphAnalyser();

  void run(Operation *op);

  bool getInConstantSubgraph(Value val);

private:
  /// Stores the result of the analysis.
  DataFlowSolver solver;

  void getConstantSubgraph(DataFlowSolver &solver, Operation *topFunc);
};
} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_CONSTANTSUBGRAPHANALYSER_H
