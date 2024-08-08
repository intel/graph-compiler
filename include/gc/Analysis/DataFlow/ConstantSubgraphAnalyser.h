//===-- ConstantSubgraphAnalyser.h - Constant subgraph ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements constant subgraph analysis. In this file are:
/// 1. the lattice value class that represents operations with constant inputs
/// and outputs in the program, and
/// 2. a sparse constant subgraph analysis.
///
///===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_CONSTANTSUBGRAPHANALYSER_H
#define MLIR_ANALYSIS_DATAFLOW_CONSTANTSUBGRAPHANALYSER_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace mlir {
namespace dataflow {

//===----------------------------------------------------------------------===//
// IsConstantTensor
//===----------------------------------------------------------------------===//

/// This lattice represents a boolean indicating if a value is constant.
class IsConstantTensor {
public:
  /// Construct as uninitialized.
  explicit IsConstantTensor() = default;

  /// Construct with a known state.
  explicit IsConstantTensor(bool initialized, bool isConstantTensor)
      : initialized(initialized), isConstantTensor(isConstantTensor) {}

  /// Get the state. Must be initialized before.
  bool getIsConstantTensor() const {
    assert(!isUninitialized());
    return isConstantTensor;
  }

  /// Compare.
  bool operator==(const IsConstantTensor &rhs) const {
    return initialized == rhs.initialized &&
           isConstantTensor == rhs.isConstantTensor;
  }

  void print(raw_ostream &os) const;

  /// Get uninitialized state. This happens when the
  /// state hasn't been set during the analysis.
  static IsConstantTensor getUninitialized() { return IsConstantTensor{}; }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !initialized; }

  /// Get unknown state.
  static IsConstantTensor getUnknown() {
    return IsConstantTensor{/*initialized=*/false,
                            /*isConstantTensor*/ false};
  }

  // Join two states.
  static IsConstantTensor join(const IsConstantTensor &lhs,
                               const IsConstantTensor &rhs) {
    // if one is uninitialized, use another
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;

    // both are initialized, intersect them
    if (!lhs.isUninitialized() && !rhs.isUninitialized()) {
      return IsConstantTensor(true, lhs.getIsConstantTensor() &&
                                        rhs.getIsConstantTensor());
    }
    return getUninitialized();
  }

private:
  bool initialized = false;
  bool isConstantTensor = false;
};

//===----------------------------------------------------------------------===//
// ConstantSubgraphAnalyser
//===----------------------------------------------------------------------===//

class ConstantSubgraphAnalyser
    : public SparseForwardDataFlowAnalysis<Lattice<IsConstantTensor>> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void visitOperation(Operation *op,
                      ArrayRef<const Lattice<IsConstantTensor> *> operands,
                      ArrayRef<Lattice<IsConstantTensor> *> results) override;

  void setToEntryState(Lattice<IsConstantTensor> *lattice) override;
};

//===----------------------------------------------------------------------===//
// RunConstantSubgraphAnalyser
//===----------------------------------------------------------------------===//

/// Runs constant subgraph analysis on the IR defined by `op`.
struct RunConstantSubgraphAnalyser {
public:
  RunConstantSubgraphAnalyser();

  void run(Operation *op);

  bool getIsConstantTensor(Value val);

private:
  /// Stores the result of the analysis.
  DataFlowSolver solver;

  void getConstantSubgraph(DataFlowSolver &solver, Operation *topFunc);
};
} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_CONSTANTSUBGRAPHANALYSER_H
