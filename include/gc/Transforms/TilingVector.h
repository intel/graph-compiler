//===- TilingVector.h - Tiling large vector to small vector ---*- C++ -*-===//
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
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <deque>
#include <iostream>
#include <queue>
#include <tuple>
#include <type_traits>
#include <variant>
namespace mlir {
namespace gc {
namespace {

Value makeIndexArithConstantOp(OpBuilder &opBuilder, Location &loc, int64_t x);
void setOperationCorrectOperand(
    Operation *op, const ValueRange &iterArgs,
    const llvm::DenseMap<Value, int> &operandIdxMap,
    ArrayRef<Value> inductionVars,
    const llvm::DenseMap<Operation *, AffineMap> &opPermuationMap);
mlir::FailureOr<Value> getOperationOperateTensor(Operation *op);

struct HardWareInfo {
  bool favx512f = true;
  bool favx2 = true;
};

/// VectorType conversion helper class
class TypeHelper {
private:
  HardWareInfo HWInfo;

public:
  void setHardWareInfo(HardWareInfo &info) { HWInfo = info; }
  int getDataTypeValidSteps(VectorType type);
  int generateValidSteps(int steps, VectorType type);
  int getDataTypeMAXSIMDLength(VectorType type);
  VectorType getVectorzedType(Operation *op, uint32_t loopStep = 0);
};

/// Operation fusion strategy class.
/// 1. Classify operaions:
/// classify the operations into :
///    a. reorder, transpose. Reorder(or transpose) dim may bring data
///    dependency.
///    b. elemenwise. Those operations can be fused into a common for loop.
///    c. broadcast. Need to analysis broadcast dim and the data
///    dependency.
///    d. reduction. Need to analysis broadcast dim and the
///    data dependency.
/// Same group operations have no data dependencies. They can be fused into a
/// common for loop body.

/// Using queue to store the operation order. In order to ensure that
/// subsequent moves to the operation will not cause semantic changes.
class VectorFusionStrategy : public TypeHelper {
private:
  func::FuncOp func;
  llvm::SmallVector<std::queue<Operation *>, 8> opGroups;
  llvm::SmallVector<uint32_t, 8> groupMaxSteps;
  /// vector type which has bigest rank in current operation group
  llvm::SmallDenseMap<size_t, VectorType> groupBigestRankVectorType;
  /// query current operation in which group, return group index
  llvm::DenseMap<Operation *, size_t> opGroupIndexMap;
  /// can fused into prev operation which axis position
  llvm::DenseMap<Operation *, size_t> opAnchorPos;

public:
  VectorFusionStrategy() = default;
  VectorFusionStrategy(func::FuncOp &func) : func(func) {}
  VectorFusionStrategy(func::FuncOp &func, TypeHelper &typeHelper)
      : TypeHelper(typeHelper), func(func) {}
  VectorFusionStrategy(VectorFusionStrategy &strategy)
      : func(strategy.func), opGroups(strategy.opGroups),
        groupMaxSteps(strategy.groupMaxSteps),
        opGroupIndexMap(strategy.opGroupIndexMap),
        opAnchorPos(strategy.opAnchorPos){};
  VectorFusionStrategy(VectorFusionStrategy &&strategy)
      : func(std::move(strategy.func)), opGroups(std::move(strategy.opGroups)),
        groupMaxSteps(std::move(strategy.groupMaxSteps)),
        opGroupIndexMap(std::move(strategy.opGroupIndexMap)),
        opAnchorPos(std::move(strategy.opAnchorPos)){};

  VectorFusionStrategy &operator=(VectorFusionStrategy &&) = default;

  llvm::SmallDenseMap<size_t, VectorType> &getGroupBiggestRankVectorType() {
    return groupBigestRankVectorType;
  };
  llvm::SmallVector<std::queue<Operation *>, 8> &getOpGroups() {
    return opGroups;
  }
  llvm::DenseMap<Operation *, size_t> &getOpGroupIndexMap() {
    return opGroupIndexMap;
  }
  llvm::SmallVector<uint32_t, 8> &getGroupMaxSteps() { return groupMaxSteps; }
  llvm::DenseMap<Operation *, size_t> &getOpAnchorPos() { return opAnchorPos; }

  func::FuncOp &getFunc() { return func; }

  void classifyOperations();

  /// Whether two operations have compatible vector shapes
  bool isCompatibleVectorType(Operation *op1, Operation *op2);

  void updateGroupBitgestVectorType(VectorType vectorType);

  /// Check whether the operation can fuse with previous operation
  bool isNeedNewGroup(Operation *op);

  /// Add Operation \p op into current last group or a new Group
  /// \p op must has valid value, can't be nullptr
  void addOperationToGroup(Operation *op);

  /// run the vector-based fusion strategy
  void run();
};

enum CanonicalizerKind { OperationsGroup, Operations };

template <class T> class SpecialOperationCanonicalizer {
private:
  llvm::SmallVector<T, 4> candidateRdOps;

public:
  enum class SpecialOperationKind {
    OP_MultiDimReduction,
    OP_Broadcast,
    OP_Transpose,
    OP_ShapeCast
  };

private:
  const SpecialOperationKind kind;

public:
  SpecialOperationCanonicalizer() = default;
  SpecialOperationCanonicalizer(const llvm::SmallVector<T, 4> &candidateRdOps,
                                SpecialOperationKind kind)
      : candidateRdOps(candidateRdOps), kind(kind) {}
  llvm::SmallVector<T, 4> &getCandidateOps();
  virtual ~SpecialOperationCanonicalizer() {}
  virtual void prepareSpecialOperationInfo() = 0;
  SpecialOperationKind getKind() { return kind; }
};

enum class MultiReduceOpAxisKind { Reduction, Parallel };
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
            candidateRdOps, SpecialOperationKind::OP_MultiDimReduction) {
    isStandaloneOp = candidateRdOps.size() == 1;
    prepareSpecialOperationInfo();
  };
  virtual ~MultiReductionCanonicalizer(){};
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
  VectorType getSourceType() { return sourceType; };
  VectorType getAccType() { return accType; };
  llvm::SmallDenseMap<Value, int> &getResultIdxMap() { return resultIdxMap; }
  void setResultIdxMap(const llvm::SmallDenseMap<Value, int> &map) {
    resultIdxMap = map;
  }

  void prepareSpecialOperationInfo() override;

  static bool classof(SpecialOperationCanonicalizer *canonicalizer) {
    return canonicalizer->getKind() ==
           SpecialOperationKind::OP_MultiDimReduction;
  }
};

class BroadcastCanonicalizer
    : public SpecialOperationCanonicalizer<vector::BroadcastOp> {
private:
public:
  BroadcastCanonicalizer(
      const llvm::SmallVector<vector::BroadcastOp, 4> &candidateBcOps)
      : SpecialOperationCanonicalizer<vector::BroadcastOp>(
            candidateBcOps, SpecialOperationKind::OP_Broadcast){};
  virtual ~BroadcastCanonicalizer() {}
  void prepareSpecialOperationInfo() override {}
  static bool classof(SpecialOperationCanonicalizer *canonicalizer) {
    return canonicalizer->getKind() == SpecialOperationKind::OP_Broadcast;
  }
};

class TransposeCanonicalizer
    : public SpecialOperationCanonicalizer<vector::TransposeOp> {
private:
public:
  TransposeCanonicalizer(
      const llvm::SmallVector<vector::TransposeOp, 4> &candidateTpOps)
      : SpecialOperationCanonicalizer<vector::TransposeOp>(
            candidateTpOps, SpecialOperationKind::OP_Transpose){};
  virtual ~TransposeCanonicalizer() {}
  void prepareSpecialOperationInfo() override {}
  static bool classof(SpecialOperationCanonicalizer *canonicalizer) {
    return canonicalizer->getKind() == SpecialOperationKind::OP_Transpose;
  }
};

class ShapeCastCanonicalizer
    : public SpecialOperationCanonicalizer<vector::ShapeCastOp> {
private:
public:
  ShapeCastCanonicalizer(
      const llvm::SmallVector<vector::ShapeCastOp, 4> &candidateScOps)
      : SpecialOperationCanonicalizer<vector::ShapeCastOp>(
            candidateScOps, SpecialOperationKind::OP_ShapeCast){};
  virtual ~ShapeCastCanonicalizer() {}
  void prepareSpecialOperationInfo() override {}
  static bool classof(SpecialOperationCanonicalizer *canonicalizer) {
    return canonicalizer->getKind() == SpecialOperationKind::OP_ShapeCast;
  }
};

enum class ReturnTypeKind {
  RT_Both,
  RT_OutGroup,
  RT_InGroup,
};

class CanonicalizerCommonUsedData : public TypeHelper {
private:
  VectorFusionStrategy fusionStrategy;

private:
  /// analysis the operation's operands and results
  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
      groupOpResults;
  llvm::SmallVector<llvm::SetVector<Value>, 8> groupOpInitArgs;

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
      llvm::SmallVector<
          llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
          &groupOpResults,
      llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOpInitArgs,
      llvm::DenseMap<Operation *, AffineMap> &opPermuationMap)
      : fusionStrategy(fusionStrategy), groupOpResults(groupOpResults),
        groupOpInitArgs(groupOpInitArgs), opPermuationMap(opPermuationMap) {}
  virtual ~CanonicalizerCommonUsedData(){};

  /// Set fusion strategy
  void setFuseStrategy(VectorFusionStrategy &&strategy) {
    fusionStrategy = std::move(strategy);
    llvm::SmallVector<std::queue<Operation *>, 8> &opGroups =
        fusionStrategy.getOpGroups();
    // init operations results and initialization args
    if (opGroups.size() != groupOpResults.size() ||
        opGroups.size() != groupOpInitArgs.size()) {
      groupOpResults.clear();
      groupOpInitArgs.clear();
      for (size_t i = 0; i < opGroups.size(); i++) {
        groupOpResults.emplace_back(
            llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>());
        groupOpInitArgs.emplace_back(llvm::SetVector<Value>());
      }
    }
  }
  void setGroupOpResults(
      const SmallVector<
          llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
          &results) {
    groupOpResults = std::move(results);
  }
  void setGroupOpIterArgs(
      const llvm::SmallVector<llvm::SetVector<Value>, 8> &initArgs) {
    groupOpInitArgs = std::move(initArgs);
  }
  void setPermutationMap(const llvm::DenseMap<Operation *, AffineMap> &map) {
    opPermuationMap = std::move(map);
  }

  // get methods
  VectorFusionStrategy &getFusionStrategy() { return fusionStrategy; }

  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8> &
  getGroupOpResults() {
    return groupOpResults;
  }

  llvm::SmallVector<llvm::SetVector<Value>, 8> &getGroupOpInitArgs() {
    return groupOpInitArgs;
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

  void generateEmptyTensorAndWrite(
      Operation *sourceOp,
      llvm::DenseMap<Operation *, std::pair<Value, Value>>
          &srcOpCanoniclizedMap,
      size_t anchorPos, ReturnTypeKind retKind);

  void updateOpOperandResultInGroups(size_t opGid, Operation *op, Value &init,
                                     const Value &result = Value());

  Value
  canonicalizeCurrentOperation(Operation *op, const Value &transferReadOperand,
                               size_t operandIdx,
                               vector::TransferReadOp *srcReadOp = nullptr);

  Operation *
  createTransferReadOpBefore(Operation *op, const Value &operand,
                             vector::TransferReadOp *srcReadOp = nullptr);
};

class ForLoopGenerator : virtual public CanonicalizerCommonUsedData {
private:
  func::FuncOp func;

public:
  ForLoopGenerator() = default;
  ForLoopGenerator(func::FuncOp &func) : func(func) {}

  virtual ~ForLoopGenerator() {}
  void setGeneratorFunc(func::FuncOp &func) { this->func = func; }
  void generateGroupOpVectorizedIR(const int idx);
  void
  rewriteOperationAsVectorize(OpBuilder &rewriter, size_t groupId,
                              const std::queue<Operation *> *queue = nullptr);
  void createNewConstantOp(Operation *srcOp,
                           vector::TransferWriteOp *transferWriteOp);
  // elementwise for loop
  mlir::FailureOr<scf::ForOp>
  generateVectorizedForLoop(const size_t groupId, IRRewriter &rewriter,
                            const VectorType vectorType);

  scf::ForOp
  constructNestedForOp(const size_t forDimIdx, const size_t groupIdx,
                       OpBuilder &b, const Location &loc,
                       const ValueRange &iterArgs, VectorType type,
                       const llvm::ArrayRef<int64_t> &dims,
                       llvm::SmallVector<Value, 5> &inductionVars,
                       const llvm::DenseMap<Value, int> &operandIdxMap);
  void moveOperationsToCurrentForBody(
      const size_t groupIdx, const OpBuilder &b, ArrayRef<Value> inductionVars,
      const llvm::DenseMap<Value, int> &operandIdxMap,
      const ValueRange &loopState, std::queue<Operation *> &queue);

  void getResultInCurrentOps(const size_t anchorIdx, const size_t groupId,
                             const std::queue<Operation *> ops,
                             SmallVector<Value, 4> &results,
                             DenseMap<Value, Value> &forResultOrignalResultMap);
  void
  getInitArgsToNextAnchor(const size_t anchorIdx, const size_t groupId,
                          const std::queue<Operation *> &nextOperations,
                          const ValueRange &loopState,
                          llvm::DenseMap<Value, int> &currentLoopStateIdxMap,
                          llvm::DenseMap<Value, int> &nextAnchorArgsIdxMap,
                          llvm::SmallVector<Value, 4> &nextAnchorArgs,
                          DenseMap<Value, Value> &originalOperandLoopArgsMap,
                          DenseMap<Value, Value> &loopArgsOriginalOperandMap);

  void getOperationInCurrentAnchor(const size_t anchorIdx,
                                   std::queue<Operation *> &fromQueue,
                                   std::queue<Operation *> &toQueue);
  void generateLoopResults(OpBuilder &b, const Location &loc,
                           const size_t anchorIdx, const size_t groupIdx,
                           llvm::SmallVector<Value, 4> &nextAnchorResults,
                           llvm::DenseMap<Value, int> &nextAnchorResultsIdxMap,
                           const ValueRange &forResults,
                           const std::queue<Operation *> &movedOperaiton,
                           DenseMap<Value, Value> &forResultOrignalResultMap);

  void movePostOpToCurrentAnchor(
      OpBuilder &b, const int anchorIdx, const int groupIdx,
      const ValueRange &forResults, const Block *forBlock,
      std::queue<Operation *> &candidateOps,
      std::queue<Operation *> &movedOperation, ArrayRef<Value> inductionVars,
      const llvm::DenseMap<Value, int> &operandIdxMap,
      const ValueRange &loopState,
      const llvm::SmallVector<Value, 4> &nextAnchorResults,
      DenseMap<Value, Value> &forResultOrignalResultMap);

  void
  movePreOpToCurrentAnchor(const size_t anchorIdx, const size_t groupIdx,
                           OpBuilder &b, ArrayRef<Value> inductionVars,
                           const ValueRange &loopState,
                           llvm::DenseMap<Value, int> &currentLoopStateIdxMap,
                           llvm::DenseMap<Value, int> &nextLoopStateIdxMap,
                           llvm::SmallVector<Value, 4> &nextAnchorArgs,
                           std::queue<Operation *> &candidateQueue,
                           std::queue<Operation *> &movedQueue,
                           DenseMap<Value, Value> &originalOperandLoopArgsMap,
                           DenseMap<Value, Value> &LoopArgsoriginalOperandMap);

  void replaceOperationsWithForLoopResult(
      IRRewriter &rewrite, const ValueRange &forResults, const Block *forBlock,
      const llvm::SmallVector<Value, 4> &nextAnchorResults,
      const std::queue<Operation *> &movingOperations,
      DenseMap<Value, Value> &forResultOrignalResultMap);
  // multireduction forloop  methods
  scf::ForOp generateMultiReductionForLoop(const size_t grpIdx);
  scf::ForOp reductionAxisGenerateForLoop(
      OpBuilder &opBuilder, const int groupIdx, const size_t reductionIdx,
      const int anchorIdx, llvm::DenseMap<Value, int> &currentLoopStateIdxMap,
      const ValueRange &initArgs,
      DenseMap<Value, Value> &originalOperandLoopArgsMap,
      DenseMap<Value, Value> &loopArgsOriginalOperandMap,
      llvm::SmallVector<Value, 4> &nextAnchorResults,
      llvm::DenseMap<Value, int> &nextAnchorResultsIdxMap,
      llvm::SmallVector<Value, 5> &inductionVars,
      DenseMap<Value, Value> &forResultOrignalResultMap,
      DenseMap<Value, Value> &originalResultForResultMap);

  scf::ForOp parallelAxisGenerateForLoop(
      OpBuilder &opBuilder, const int groupIdx, const size_t parallelIdx,
      llvm::DenseMap<Value, int> &currentLoopStateIdxMap,
      const ValueRange &initArgs,
      llvm::SmallVector<Value, 4> &nextAnchorResults,
      llvm::DenseMap<Value, int> &nextAnchorResultsIdxMap,
      llvm::SmallVector<Value, 5> &inductionVars,
      DenseMap<Value, Value> &originalOperandLoopArgsMap,
      DenseMap<Value, Value> &loopArgsOriginalOperandMap,
      DenseMap<Value, Value> &forResultOrignalResultMap);

  vector::TransferReadOp cloneReductionTransferRead(
      Value &source, OpBuilder &b, IRMapping &readMap,
      const llvm::SmallVector<int64_t, 4> &parallelAxis,
      llvm::SmallVector<Value, 5> &inductionVars, bool lastDimReduction,
      MultiReduceOpAxisKind rdKind = MultiReduceOpAxisKind::Parallel);
};

class VectorOperationAnalyzer : virtual public CanonicalizerCommonUsedData {
private:
  func::FuncOp func;

public:
  virtual ~VectorOperationAnalyzer(){};
  VectorOperationAnalyzer() {}
  VectorOperationAnalyzer(func::FuncOp &func) : func(func) {}

  void setAnalysisFunc(func::FuncOp &func) { this->func = func; }
  void analysisEmptyGroupAndMaxSteps();
  void analysisGroupOperaion();
  void analysisGroupOperationResults();
  void specialOperationAnchorRectify();
};
/// Vectorize vector operation with target machines simd instructions.
class CanonicalizerVectorOperation : virtual public ForLoopGenerator,
                                     VectorOperationAnalyzer {
private:
  func::FuncOp func;
  IRRewriter rewriter;

  CanonicalizerKind kind;

public:
  CanonicalizerVectorOperation(
      func::FuncOp func,
      CanonicalizerKind kind = CanonicalizerKind::OperationsGroup,
      HardWareInfo hwInfo = {})
      : func(func), rewriter(func), kind(kind) {
    setAnalysisFunc(func);
    setGeneratorFunc(func);
    setHardWareInfo(hwInfo);
    // vector operation fusion
    if (kind == CanonicalizerKind::OperationsGroup) {
      VectorFusionStrategy fusionStrategy(func);
      fusionStrategy.run();
      setFuseStrategy(std::move(fusionStrategy));
    }
  }
  virtual ~CanonicalizerVectorOperation(){};

  // get functions
  func::FuncOp &getFunc() { return func; };
  IRRewriter &getIRWewriter() { return rewriter; }
  //
  void canonicalizeSpecialOperation();
  void clearSpecialOperationCanonicalizers();
  void dummyInitSpecialOperation();
  void initSpeicalOperationCanonicalizers();

  void run();
};
} // namespace
} // namespace gc
} // namespace mlir
#endif