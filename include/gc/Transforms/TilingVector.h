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
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
namespace mlir {
namespace gc {
namespace {

Value makeIndexArithConstantOp(OpBuilder &opBuilder, Location &loc, int64_t x);
void setOperationCorrectOperand(
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
  llvm::SmallVector<uint32_t, 8> groupMaxSteps;
  // query current operation in which group, return group index
  llvm::DenseMap<Operation *, size_t> opGroupIndexMap;
  // can fused into prev operation which axis position
  llvm::DenseMap<Operation *, int32_t> opAnchorPos;

  func::FuncOp func;

public:
  llvm::SmallVector<std::queue<Operation *>, 8> &getOpGroups() {
    return opGroups;
  }
  llvm::DenseMap<Operation *, size_t> &getOpGroupIndexMap() {
    return opGroupIndexMap;
  }
  llvm::SmallVector<uint32_t, 8> &getGroupMaxSteps() { return groupMaxSteps; }

  func::FuncOp getFunc() { return func; }

  VectorFusionStrategy() = default;
  VectorFusionStrategy(func::FuncOp func) : func(func) {}

  void classifyOperations();

  // run the vector fusion strategy
  void run();
};

enum CanonicalizerKind { OperationsGroup, Operations };

template <class T> class SpecialOperationCanonicalizer {
private:
  llvm::SmallVector<T, 4> candidateRdOps;

public:
  SpecialOperationCanonicalizer() = default;
  SpecialOperationCanonicalizer(const llvm::SmallVector<T, 4> &candidateRdOps)
      : candidateRdOps(candidateRdOps) {}
  llvm::SmallVector<T, 4> &getCandidateOps();
  virtual void prepareSpecialOperationInfo() = 0;
};

class MultiReductionCanonicalizer
    : public SpecialOperationCanonicalizer<vector::MultiDimReductionOp> {
private:
  llvm::SmallVector<int64_t, 4> reductionAxis, parallelAxis;
  std::queue<Operation *> prevOps, postOps, accRelatedOps, sourceRelatedOps;
  bool haslastDimReduction = false;
  bool isStandaloneOp = false;
  int64_t typeRank = -1;
  llvm::SetVector<Value> originalOpResults;
  VectorType sourceType, accType;
  llvm::SmallDenseMap<Value, int> resultIdxMap;

public:
  MultiReductionCanonicalizer(
      const llvm::SmallVector<vector::MultiDimReductionOp, 4> &candidateRdOps)
      : SpecialOperationCanonicalizer<vector::MultiDimReductionOp>(
            candidateRdOps) {
    isStandaloneOp = candidateRdOps.size() == 1;
    prepareSpecialOperationInfo();
  };
  int64_t getTypeRank();
  void getReductionAxisAndParallelAxis();
  bool hasLastDimReduction();
  bool getIsStandAloneOp() { return isStandaloneOp; }
  void initReductionAxis();
  void initParallelAxis();
  llvm::SmallVector<int64_t, 4> &getReductionAxis() { return reductionAxis; };
  llvm::SmallVector<int64_t, 4> &getParallelAxis() { return parallelAxis; };
  std::queue<Operation *> &getPrevOps() { return prevOps; }
  std::queue<Operation *> &getPostOps() { return postOps; }
  std::queue<Operation *> &getAccRelatedOps() { return accRelatedOps; }
  std::queue<Operation *> &getSourceRelatedOps() { return sourceRelatedOps; }
  llvm::SetVector<Value> &getOriginalOpResults() { return originalOpResults; }
  VectorType &getSourceType() { return sourceType; };
  VectorType &getAccType() { return accType; };
  llvm::SmallDenseMap<Value, int> &getResultIdxMap() { return resultIdxMap; }
  void setResultIdxMap(const llvm::SmallDenseMap<Value, int> &map) {
    resultIdxMap = std::move(map);
  }
  void prepareSpecialOperationInfo() override;
};

class BroadcastCanonicalizer
    : public SpecialOperationCanonicalizer<vector::BroadcastOp> {
private:
public:
  BroadcastCanonicalizer(
      const llvm::SmallVector<vector::BroadcastOp, 4> &candidateBcOps)
      : SpecialOperationCanonicalizer<vector::BroadcastOp>(candidateBcOps){};
  void prepareSpecialOperationInfo() override {}
};

class TransposeCanonicalizer
    : public SpecialOperationCanonicalizer<vector::TransposeOp> {
private:
public:
  TransposeCanonicalizer(
      const llvm::SmallVector<vector::TransposeOp, 4> &candidateTpOps)
      : SpecialOperationCanonicalizer<vector::TransposeOp>(candidateTpOps){};
  void prepareSpecialOperationInfo() override {}
};

class ShapeCastCanonicalizer
    : public SpecialOperationCanonicalizer<vector::ShapeCastOp> {
private:
public:
  ShapeCastCanonicalizer(
      const llvm::SmallVector<vector::ShapeCastOp, 4> &candidateScOps)
      : SpecialOperationCanonicalizer<vector::ShapeCastOp>(candidateScOps){};
  void prepareSpecialOperationInfo() override {}
};

class CanonicalizerCommonUsedData {
private:
  VectorFusionStrategy fusionStrategy;
  // analysis the operation's operands and results
  llvm::SmallVector<llvm::SetVector<Value>, 8> groupOpResults, groupOpIterArgs;

  // store read and write operations permutation maps in order to convenient
  // to replace loop induction var
  llvm::DenseMap<Operation *, AffineMap> opPermuationMap;
  llvm::SmallVector<MultiReductionCanonicalizer, 8> multiRdCanonicalizers;
  llvm::SmallVector<BroadcastCanonicalizer, 8> broadcastCanonicalizers;
  llvm::SmallVector<TransposeCanonicalizer, 8> transposeCanonicalizers;
  llvm::SmallVector<ShapeCastCanonicalizer, 8> shapeCastCanonicalizers;

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
  virtual ~CanonicalizerCommonUsedData(){};

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

  llvm::SmallVector<MultiReductionCanonicalizer, 8> &
  getMultiRdCanonicalizers() {
    return multiRdCanonicalizers;
  }

  llvm::SmallVector<BroadcastCanonicalizer, 8> &getBroadcastCanonicalizers() {
    return broadcastCanonicalizers;
  }

  llvm::SmallVector<TransposeCanonicalizer, 8> &getTransposeCanonicalizers() {
    return transposeCanonicalizers;
  }

  llvm::SmallVector<ShapeCastCanonicalizer, 8> &getShapeCastCanonicalizers() {
    return shapeCastCanonicalizers;
  }

  // other methods
  bool isGroupHasSpecialOperation(const size_t grpIdx);
};

class ForLoopGenerator : virtual public CanonicalizerCommonUsedData {
  func::FuncOp func;

public:
  virtual ~ForLoopGenerator() {}
  void setGeneratorFunc(func::FuncOp &func) { this->func = func; }
  void generateGroupOpVectorizedIR(const int idx);
  void rewriteOperationAsVectorize(OpBuilder &rewriter, size_t groupId,
                                   const std::queue<Operation *> &queue = {});
  void createNewConstantOp(Operation *srcOp,
                           vector::TransferWriteOp *transferWriteOp);
  // elementwise for loop
  mlir::FailureOr<scf::ForOp>
  generateVectorizedForLoop(const size_t groupId, IRRewriter &rewriter,
                            const VectorType &vectorType);
  scf::ForOp
  constructNestedForOp(const size_t forDimIdx, const size_t groupIdx,
                       OpBuilder &b, const Location &loc,
                       const ValueRange &iterArgs, const VectorType &type,
                       const llvm::ArrayRef<int64_t> &dims,
                       llvm::SmallVector<Value, 5> &inductionVars,
                       const llvm::DenseMap<Value, int> &operandIdxMap);
  void moveOperationsToCurrentForBody(
      const size_t groupIdx, OpBuilder &b,
      const llvm::SmallVector<Value, 5> &inductionVars,
      const llvm::DenseMap<Value, int> &operandIdxMap,
      const ValueRange &loopState, const std::queue<Operation *> &queue = {});

  // multireduction forloop  methods
  scf::ForOp generateMultiReductionForLoop(const size_t grpIdx);
  scf::ForOp
  parallelAxisGenerateForLoop(OpBuilder &opBuilder, const int groupIdx,
                              const size_t parallelIdx, ValueRange &initArgs,
                              llvm::SmallVector<Value, 5> &inductionVars,
                              Value &originalWriteResult);

  scf::ForOp
  reductionAxisGenerateForLoop(OpBuilder &opBuilder, const int groupIdx,
                               const size_t reductionIdx, ValueRange &initArgs,
                               llvm::SmallVector<Value, 5> &inductionVars);
};

class VectorOperationAnalysizer : virtual public CanonicalizerCommonUsedData {
private:
  func::FuncOp func;

public:
  virtual ~VectorOperationAnalysizer(){};
  void generateEmptyTensorAndWrite(
      Operation *sourceOp, llvm::DenseMap<Operation *, std::pair<Value, Value>>
                               &srcOpCanoniclizedMap);
  void setAnalysisFunc(func::FuncOp &func) { this->func = func; }
  void analysisEmptyGroupAndMaxSteps();
  void analysisGroupOperaion();
  void analysisGroupOperationResults();
};

class CanonicalizerVectorOperation : virtual public VectorOperationAnalysizer,
                                     ForLoopGenerator {
private:
  func::FuncOp func;
  IRRewriter rewriter;
  CanonicalizerKind kind;

public:
  CanonicalizerVectorOperation(
      func::FuncOp func,
      CanonicalizerKind kind = CanonicalizerKind::OperationsGroup)
      : func(func), rewriter(func), kind(kind) {
    setAnalysisFunc(func);
    setGeneratorFunc(func);
    // vector operation fusion
    if (kind == CanonicalizerKind::OperationsGroup) {
      auto fusionStrategy = VectorFusionStrategy(func);
      fusionStrategy.run();
      setFuseStrategy(fusionStrategy);
    }
  }
  virtual ~CanonicalizerVectorOperation(){};

  // get functions
  func::FuncOp &getFunc() { return func; };
  IRRewriter &getIRWewriter() { return rewriter; }
  //
  void canonicalizeSpecialOperation();
  LogicalResult canonicalizeReductionOperation();
  void clearSpecialOperationCanonicalizers();
  void dummyInitSpecialOperation();
  void initSpeicalOperationCanonicalizers();

  void run();
};
} // namespace
} // namespace gc
} // namespace mlir
#endif