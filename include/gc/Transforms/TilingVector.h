//===- TilingVector.h - Graph Compiler passes -------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef GC_PASSES_TILINGVECTOR_H
#define GC_PASSES_TILINGVECTOR_H

#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <deque>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <tuple>
#include <utility>
#include <variant>
namespace mlir {
namespace gc {
namespace {

Value makeIndexArithConstantOp(OpBuilder &opBuilder, Location &loc, int64_t x);
void rewriteOperationAsVectorize(
    const std::queue<Operation *> &groupOps,
    const llvm::DenseMap<Operation *, size_t> &opMap, OpBuilder &rewriter,
    llvm::DenseMap<Operation *, AffineMap> &opPermuationMap);
void checkAndSetOperand(
    Operation *op, const ValueRange &iterArgs,
    const llvm::DenseMap<Value, int> &operandIdxMap,
    const llvm::SmallVector<Value, 5> &inductionVars,
    const llvm::DenseMap<Operation *, AffineMap> &opPermuationMap);

// 1. Classify operaions:
// classify the operations into :
//    a. reorder, transpose. Reorder(or transpose) dim may bring data
//    dependency.
//    b. elemenwise. Those operations can be fused into a common for loop.
//    c. broadcast. Need to analysis broadcast dim and the data
//    dependency.
//    d. reduction. Need to analysis broadcast dim and the
//    data dependency.
// Same group operations have no data dependencies. They can be fused into a
// common for loop body.

// Using queue to store the operation order. In order to ensure that
// subsequent moves to the operation will not cause semantic changes.
class VectorFusionStrategy {
private:
  llvm::SmallVector<std::queue<Operation *>, 8> opGroups;
  // query current operation in which group, return group index
  llvm::DenseMap<Operation *, size_t> opGroupIndexMap;
  // can fused into prev operation which axis position
  llvm::DenseMap<Operation *, int32_t> opAnchorPos;

  llvm::SmallVector<std::queue<Operation *>, 8> ignoreInitOperations;

  func::FuncOp func;

public:
  llvm::SmallVector<std::queue<Operation *>, 8> &getOpGroups() {
    return opGroups;
  }
  llvm::DenseMap<Operation *, size_t> &getOpGroupIndexMap() {
    return opGroupIndexMap;
  }

  func::FuncOp getFunc() { return func; }
  llvm::SmallVector<std::queue<Operation *>, 8> getIgnoreInitOperations() {
    return ignoreInitOperations;
  }

  VectorFusionStrategy() = default;
  VectorFusionStrategy(func::FuncOp func) : func(func) {}

  void classifyOperations();

  // run the vector fusion strategy
  void run();
};

enum CanonicalizerKind { OperationsGroup, Operations };

class MultiReductionCanonicalizer {
private:
  llvm::SmallVector<vector::MultiDimReductionOp, 4> candidateRdOps;
  llvm::SmallVector<int64_t, 4> reductionAxis, parallelAxis;
  bool haslastDimReduction = false;
  bool isStandaloneOp = false;
  int64_t typeRank = -1;

public:
  MultiReductionCanonicalizer(
      const llvm::SmallVector<vector::MultiDimReductionOp, 4> &candidateRdOps)
      : candidateRdOps(candidateRdOps) {
    assert(candidateRdOps.size() > 1);
    isStandaloneOp = candidateRdOps.size() == 1;
    prepareReductionInfo();
  };
  int64_t getTypeRank();
  llvm::SmallVector<vector::MultiDimReductionOp, 4> &getCandidateOps();
  void getReductionAxisAndParallelAxis();
  bool hasLastDimReduction();
  bool getIsStandAloneOp() { return isStandaloneOp; }
  void initReductionAxis();
  void initParallelAxis();
  llvm::SmallVector<int64_t, 4> &getReductionAxis() { return reductionAxis; };
  llvm::SmallVector<int64_t, 4> &getParallelAxis() { return parallelAxis; };
  void prepareReductionInfo();
};

class CanonicalizerCommonUsedData {
private:
  VectorFusionStrategy fusionStrategy;
  // analysis the operation's operands and results
  llvm::SmallVector<llvm::SetVector<Value>, 8> groupOpResults, groupOpIterArgs;

  // store read and write operations permutation maps in order to convenient
  // to replace loop induction var
  llvm::DenseMap<Operation *, AffineMap> opPermuationMap;
  llvm::SmallVector<MultiReductionCanonicalizer, 8> multiRdCanonicalizer;

public:
  CanonicalizerCommonUsedData() = default;
  CanonicalizerCommonUsedData(VectorFusionStrategy &fusionStrategy)
      : fusionStrategy(fusionStrategy){};

  CanonicalizerCommonUsedData(
      VectorFusionStrategy &fusionStrategy,
      llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOpResults,
      llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOpIterArgs,
      llvm::DenseMap<Operation *, AffineMap> &opPermuationMap)
      : fusionStrategy(fusionStrategy), groupOpResults(groupOpResults),
        groupOpIterArgs(groupOpIterArgs), opPermuationMap(opPermuationMap) {}

  // set methods
  void setFuseStrategy(VectorFusionStrategy &strategy) {
    fusionStrategy = strategy;
    auto opGroups = fusionStrategy.getOpGroups();
    if (opGroups.size() != groupOpResults.size() ||
        opGroups.size() != groupOpIterArgs.size()) {
      groupOpResults.clear();
      groupOpIterArgs.clear();
      for (size_t i = 0; i < opGroups.size(); i++) {
        groupOpResults.emplace_back(llvm::SetVector<Value>());
        groupOpIterArgs.emplace_back(llvm::SetVector<Value>());
      }
    }
  }
  void
  setGroupOpResults(llvm::SmallVector<llvm::SetVector<Value>, 8> &results) {
    groupOpResults = results;
  }
  void
  setGroupOpIterArgs(llvm::SmallVector<llvm::SetVector<Value>, 8> &iterArgs) {
    groupOpIterArgs = iterArgs;
  }
  void setPermutationMap(llvm::DenseMap<Operation *, AffineMap> &map) {
    opPermuationMap = map;
  }

  // get methods
  VectorFusionStrategy &getFusionStrategy() { return fusionStrategy; }

  llvm::SmallVector<llvm::SetVector<Value>, 8> &getGroupOpResults() {
    return groupOpResults;
  }

  llvm::SmallVector<llvm::SetVector<Value>, 8> &getGroupOpIterArgs() {
    return groupOpIterArgs;
  }

  llvm::DenseMap<Operation *, AffineMap> &getOpPermuationMap() {
    return opPermuationMap;
  }
  llvm::SmallVector<MultiReductionCanonicalizer, 8> &getMultiRdCanonicalizer() {
    return multiRdCanonicalizer;
  }
};

class CanonicalizerVectorOperation {
private:
  func::FuncOp func;
  IRRewriter rewriter;
  CanonicalizerKind kind;
  CanonicalizerCommonUsedData commonUsedData;

public:
  CanonicalizerVectorOperation(
      func::FuncOp func,
      CanonicalizerKind kind = CanonicalizerKind::OperationsGroup)
      : func(func), rewriter(func), kind(kind) {
    // vector operation fusion
    if (kind == CanonicalizerKind::OperationsGroup) {
      auto fusionStrategy = VectorFusionStrategy(func);
      fusionStrategy.run();
      commonUsedData.setFuseStrategy(fusionStrategy);
    }
  }

  // get functions
  func::FuncOp &getFunc() { return func; };
  IRRewriter &getIRWewriter() { return rewriter; }
  CanonicalizerCommonUsedData &getCommonUsedData() { return commonUsedData; }

  void generateGroupOpVectorizedIR(const int idx);

  void analysisGroupOperaionOperandsResults();

  void analysisGroupOperationResults();

  LogicalResult canonicalizeReductionOperation();
  LogicalResult canonicalizeTransposeOperation(vector::TransposeOp &transposeOp,
                                               IRRewriter &rewriter);
  void rewriteOperationAsVectorize(OpBuilder &rewriter, size_t groupId);

  // special operation methods
  scf::ForOp generateMultiReductionForLoop(const size_t grpIdx);
  void getCandidateSpecialOps();
  void canonicalizeSpecialOperation();
  scf::ForOp parallelAxisGenerateForLoop(
      const int groupIdx, const int parallelIdx, ValueRange &initArgs,
      llvm::SmallVector<Value, 5> &inductionVars, Value &originalWriteResult);
  scf::ForOp
  reductionAxisGenerateForLoop(const int groupIdx, const size_t reductionIdx,
                               ValueRange &initArgs,
                               llvm::SmallVector<Value, 5> &inductionVars);

  void run();
};
} // namespace
} // namespace gc
} // namespace mlir
#endif