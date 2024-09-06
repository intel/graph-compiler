//===- TilingVector.h - Tiling large vector to small vector -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef GC_PASSES_TILINGVECTOR_H
#define GC_PASSES_TILINGVECTOR_H

#include "gc/Analysis/TargetDescriptionAnalysis.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/IR/AffineExpr.h"
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
#include <queue>
#include <stack>
#include <tuple>
#include <type_traits>
#include <variant>
// #include "gc/Dialect/Microkernel/MicrokernelOps.h"
namespace mlir {
namespace gc {
namespace {

//===----------------------------------------------------------------------===//
// helper function
//===----------------------------------------------------------------------===//

/// build a constant operation of index type
Value makeIndexArithConstantOp(OpBuilder &opBuilder, Location &loc, int64_t x);
/// set correct operand for the operation
void setOperationCorrectOperand(
    Operation *op, ValueRange iterArgs, DenseMap<Value, int> &operandIdxMap,
    DenseMap<Value, Value> &originalOperandLoopArgsMap,
    ArrayRef<Value> inductionVars,
    DenseMap<Operation *, AffineMap> &opPermuationMap);
/// get operation read or write tensor
mlir::FailureOr<Value> getOperationOperateTensor(Operation *op);

struct HardWareInfo {
  bool favx512f = true;
  bool favx2 = true;
};

//===----------------------------------------------------------------------===//
// helper function
//===----------------------------------------------------------------------===//
/// Using to avoid too many parameters in function
struct GenerateLoopHelper {
  /// anchor id
  size_t anchorIdx = 0;
  /// group id
  size_t groupIdx = 0;
  /// for loop results
  ValueRange forResults;
  /// for loop block
  Block *forBlock;
  /// loop iteration args index map
  DenseMap<Value, int> currentLoopStateIdxMap;
  /// loop iteration args
  ValueRange loopIterArgs;
  /// next loop anchor yield results
  SmallVector<Value, 4> nextAnchorResults;
  /// next loop anchor yield results index map
  DenseMap<Value, int> nextAnchorResultsIdxMap;
  /// next loop anchor yield results original result map
  DenseMap<Value, Value> nextAnchorResultOrignalResultMap;
  /// original result with next anchor result map
  DenseMap<Value, Value> orignalResultNextAnchorResultMap;
  /// loop induction variables
  SmallVector<Value, 5> inductionVars;
  /// original operand with loop args map
  DenseMap<Value, Value> originalOperandLoopArgsMap;
  /// loop args with original operand map
  DenseMap<Value, Value> loopArgsOriginalOperandMap;
  /// candidate operation queue
  std::queue<Operation *> *candidateOps;
  /// moved operation queue
  std::queue<Operation *> *movedOps;
  /// record operation's correct loop indice, due to some operation like
  /// reduce may need to reorder loop indice
  DenseMap<Operation *, DenseMap<size_t, size_t>> indiceLoopMap;
  GenerateLoopHelper() = default;
  GenerateLoopHelper(const size_t groupIdx) noexcept {
    this->groupIdx = groupIdx;
  }
  GenerateLoopHelper(const size_t groupIdx, const size_t anchorIdx) noexcept {
    this->groupIdx = groupIdx;
    this->anchorIdx = anchorIdx;
  }
  /// clear next anchor results related data
  void clearNextAnchorResults();
  /// set next anchor results related data
  void setNextAnchorResults(SmallVector<Value> &currentAnchorResults,
                            DenseMap<Value, Value> &currentResultMap,
                            DenseMap<Value, int> &currentResultIdxMap);
  /// set next anchor iteration args
  void setNextAnchorArgs(DenseMap<Value, int> &nextAnchorArgsIdxMap,
                         SmallVector<Value, 4> &nextAnchorArgs);
  /// set id of for loop anchor
  void setAnchorId(const size_t anchorId) noexcept;
  /// Before perform processing previous operation, we need to update some data
  void updateDataBeforePreOpMove(ArrayRef<Value> loopstate,
                                 std::queue<Operation *> &candidateQueue,
                                 std::queue<Operation *> &movedQueue);
  /// After previous operation movement, we need to update some data
  void updateDataAfterPreOpMove(DenseMap<Value, int> &nextAnchorArgsIdxMap,
                                SmallVector<Value, 4> &nextAnchorArgs);
  /// Before perform processing previous operation, we need to update some data
  void updateDataBeforePostOpMove(
      ArrayRef<Value> iterArgs, DenseMap<Value, int> &currentLoopStateIdxMap,
      DenseMap<Value, Value> &currentoriginalArgsMap,
      DenseMap<Value, Value> &currentArgsOriginalMap, ValueRange forResults,
      Block *forBlock, std::queue<Operation *> &movedQueue, size_t anchorId);
  /// After previous operation movement, we need to update some data
  void updateDataAfterPostOpMove(size_t anchorId,
                                 DenseMap<Value, int> &nextAnchorArgsIdxMap,
                                 SmallVector<Value, 4> &nextAnchorArgs);

  void updateCurrentArgsStatus(DenseMap<Value, int> &currentArgsIdxMap,
                               SmallVector<Value, 4> &currentArgs,
                               DenseMap<Value, Value> &originalArgsMap,
                               DenseMap<Value, Value> &argsOriginalMap);
};

void GenerateLoopHelper::setNextAnchorArgs(
    DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs) {
  currentLoopStateIdxMap = nextAnchorArgsIdxMap;
  loopIterArgs = nextAnchorArgs;
}

void GenerateLoopHelper::clearNextAnchorResults() {
  nextAnchorResults.clear();
  nextAnchorResultsIdxMap.clear();
  nextAnchorResultOrignalResultMap.clear();
}

void GenerateLoopHelper::setAnchorId(size_t anchorId) noexcept {
  anchorIdx = anchorId;
}

void GenerateLoopHelper::updateDataBeforePreOpMove(
    ArrayRef<Value> loopState, std::queue<Operation *> &candidateQueue,
    std::queue<Operation *> &movedQueue) {
  loopIterArgs = loopState;
  candidateOps = &candidateQueue;
  movedOps = &movedQueue;
}

void GenerateLoopHelper::updateDataAfterPreOpMove(
    DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs) {
  setNextAnchorArgs(nextAnchorArgsIdxMap, nextAnchorArgs);
}

void GenerateLoopHelper::updateDataBeforePostOpMove(
    ArrayRef<Value> iterArgs, DenseMap<Value, int> &currentLoopStateIdxMap,
    DenseMap<Value, Value> &currentoriginalArgsMap,
    DenseMap<Value, Value> &currentArgsOriginalMap, ValueRange forResults,
    Block *forBlock, std::queue<Operation *> &movedQueue, size_t anchorId) {
  this->originalOperandLoopArgsMap = currentoriginalArgsMap;
  this->loopArgsOriginalOperandMap = currentArgsOriginalMap;
  this->forResults = forResults;
  this->forBlock = forBlock;
  this->anchorIdx = anchorId;
  this->currentLoopStateIdxMap = currentLoopStateIdxMap;
  this->loopIterArgs = iterArgs;
  this->movedOps = &movedQueue;
}

void GenerateLoopHelper::updateDataAfterPostOpMove(
    size_t anchorId, DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs) {
  setAnchorId(anchorId);
  setNextAnchorArgs(nextAnchorArgsIdxMap, nextAnchorArgs);
}

void GenerateLoopHelper::setNextAnchorResults(
    SmallVector<Value> &currentAnchorResults,
    DenseMap<Value, Value> &currentResultMap,
    DenseMap<Value, int> &currentResultIdxMap) {
  nextAnchorResults = std::move(currentAnchorResults);
  nextAnchorResultOrignalResultMap = std::move(currentResultMap);
  nextAnchorResultsIdxMap = std::move(currentResultIdxMap);
}

void GenerateLoopHelper::updateCurrentArgsStatus(
    DenseMap<Value, int> &currentArgsIdxMap, SmallVector<Value, 4> &currentArgs,
    DenseMap<Value, Value> &originalArgsMap,
    DenseMap<Value, Value> &argsOriginalMap) {
  setNextAnchorArgs(currentArgsIdxMap, currentArgs);
  originalOperandLoopArgsMap = originalArgsMap;
  loopArgsOriginalOperandMap = argsOriginalMap;
}

/// Vector type conversion helper class
class TypeHelper {
private:
  HardWareInfo HWInfo;

public:
  /// use \param info to set hardware information
  void setHardWareInfo(HardWareInfo &info) { HWInfo = info; }
  /// get vector \param type max loop step according to hardware information
  int getDataTypeValidSteps(VectorType type);
  /// get vector \param type an even for loop step
  int generateValidSteps(int steps, VectorType type);
  /// get vector \param type max simd length according to hardware information
  int getDataTypeMAXSIMDLength(VectorType type);
  /// get operation's vector type
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
  SmallVector<std::queue<Operation *>, 8> opGroups;
  SmallVector<uint32_t, 8> groupMaxSteps;
  /// vector type which has bigest rank in current operation group
  llvm::SmallDenseMap<size_t, VectorType> groupBigestRankVectorType;
  /// query current operation in which group, return group index
  DenseMap<Operation *, size_t> opGroupIndexMap;
  /// can fused into prev operation which axis position
  DenseMap<Operation *, size_t> opAnchorPos;
  /// record some operations which not need to No need to judge whether can be
  /// fused
  std::queue<Operation *> noNeedToJudgeOps;

public:
  VectorFusionStrategy() = default;
  VectorFusionStrategy(func::FuncOp &func) : func(func) {}
  VectorFusionStrategy(func::FuncOp &func, TypeHelper &typeHelper)
      : TypeHelper(typeHelper), func(func) {}

  VectorFusionStrategy(VectorFusionStrategy &strategy)
      : func(strategy.func), opGroups(strategy.opGroups),
        groupMaxSteps(strategy.groupMaxSteps),
        opGroupIndexMap(strategy.opGroupIndexMap),
        opAnchorPos(strategy.opAnchorPos) {};

  VectorFusionStrategy(VectorFusionStrategy &&strategy)
      : func(std::move(strategy.func)), opGroups(std::move(strategy.opGroups)),
        groupMaxSteps(std::move(strategy.groupMaxSteps)),
        opGroupIndexMap(std::move(strategy.opGroupIndexMap)),
        opAnchorPos(std::move(strategy.opAnchorPos)) {};

  VectorFusionStrategy &operator=(VectorFusionStrategy &&) = default;

  /// Get the map which contains each group vector type which has biggest rank.
  llvm::SmallDenseMap<size_t, VectorType> &
  getGroupBiggestRankVectorType() noexcept {
    return groupBigestRankVectorType;
  };
  /// Get the operation group obtained by fusion strategy analysis
  SmallVector<std::queue<Operation *>, 8> &getOpGroups() noexcept {
    return opGroups;
  }
  /// Get the operation belong to which group index map
  DenseMap<Operation *, size_t> &getOpGroupIndexMap() noexcept {
    return opGroupIndexMap;
  }
  /// Get the map contains max steps of each group
  llvm::SmallVector<uint32_t, 8> &getGroupMaxSteps() noexcept {
    return groupMaxSteps;
  }
  llvm::DenseMap<Operation *, size_t> &getOpAnchorPos() noexcept {
    return opAnchorPos;
  }

  func::FuncOp &getFunc() { return func; }
  /// Do fusion strategy
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

/// Has two kind:
/// 1. OperationGroup:
///     The operation is converted into physical registers through our fusion
///     strategy.
/// 2. Operations:(TODO:)
///     The user ensures that there is no data dependency between operations,
///     and we directly convert the operations into physical register sizes.
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
  SmallVector<int64_t, 4> reductionAxis, parallelAxis;
  std::queue<Operation *> prevOps, postOps, accRelatedOps, sourceRelatedOps;
  bool haslastDimReduction = false;
  bool isStandaloneOp = false;
  /// empty reduction means that all the reduction axis is 1
  bool isEmptyReduction = true;
  int64_t typeRank = -1;
  SetVector<Value> originalOpResults;
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
  virtual ~MultiReductionCanonicalizer() noexcept {};
  int64_t getTypeRank();
  void getReductionAxisAndParallelAxis();
  bool hasLastDimReduction();
  bool getIsStandAloneOp() noexcept { return isStandaloneOp; }
  bool getHasLastDimReduction() noexcept { return haslastDimReduction; }
  bool getIsEmptyReduction() noexcept { return isEmptyReduction; }
  void initReductionAxis();
  void initParallelAxis();
  SmallVector<int64_t, 4> &getReductionAxis() noexcept {
    return reductionAxis;
  };
  SmallVector<int64_t, 4> &getParallelAxis() noexcept { return parallelAxis; };
  std::queue<Operation *> &getPrevOps() noexcept { return prevOps; }
  std::queue<Operation *> &getPostOps() noexcept { return postOps; }
  std::queue<Operation *> &getAccRelatedOps() noexcept { return accRelatedOps; }
  std::queue<Operation *> &getSourceRelatedOps() noexcept {
    return sourceRelatedOps;
  }
  SetVector<Value> &getOriginalOpResults() noexcept {
    return originalOpResults;
  }
  VectorType getSourceType() noexcept { return sourceType; };
  VectorType getAccType() noexcept { return accType; };
  llvm::SmallDenseMap<Value, int> &getResultIdxMap() noexcept {
    return resultIdxMap;
  }
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
            candidateBcOps, SpecialOperationKind::OP_Broadcast) {};
  virtual ~BroadcastCanonicalizer() noexcept {}
  void prepareSpecialOperationInfo() override {}
  static bool classof(SpecialOperationCanonicalizer *canonicalizer) {
    return canonicalizer->getKind() == SpecialOperationKind::OP_Broadcast;
  }
};

class TransposeCanonicalizer
    : public SpecialOperationCanonicalizer<vector::TransposeOp> {
private:
  size_t firstTpIdx = 0, secondTpIdx = 0;

public:
  TransposeCanonicalizer(
      const llvm::SmallVector<vector::TransposeOp, 4> &candidateTpOps)
      : SpecialOperationCanonicalizer<vector::TransposeOp>(
            candidateTpOps, SpecialOperationKind::OP_Transpose) {};
  virtual ~TransposeCanonicalizer() noexcept {}
  void prepareSpecialOperationInfo() override;
  static bool classof(SpecialOperationCanonicalizer *canonicalizer) {
    return canonicalizer->getKind() == SpecialOperationKind::OP_Transpose;
  }
  enum TRANSPOSE_KERNEL {
    KERNEL_16X16 = 16,
  };

  size_t getFirstTpIdx() noexcept { return firstTpIdx; }
  size_t getSecondTpIdx() noexcept { return secondTpIdx; }
  bool isTwoDTranspose();
  bool isTransposeOnAllOneDim();
};

class ShapeCastCanonicalizer
    : public SpecialOperationCanonicalizer<vector::ShapeCastOp> {
private:
public:
  ShapeCastCanonicalizer(
      const SmallVector<vector::ShapeCastOp, 4> &candidateScOps)
      : SpecialOperationCanonicalizer<vector::ShapeCastOp>(
            candidateScOps, SpecialOperationKind::OP_ShapeCast) {};
  virtual ~ShapeCastCanonicalizer() {}
  void prepareSpecialOperationInfo() override {}
  static bool classof(SpecialOperationCanonicalizer *canonicalizer) {
    return canonicalizer->getKind() == SpecialOperationKind::OP_ShapeCast;
  }
  bool isReadWriteOnLastDim();
};

/// operation return kind, which is used to determine whether the operation need
/// to return it's result in current for loop
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
      : fusionStrategy(fusionStrategy) {};

  CanonicalizerCommonUsedData(
      VectorFusionStrategy &fusionStrategy,
      llvm::SmallVector<
          llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
          &groupOpResults,
      llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOpInitArgs,
      llvm::DenseMap<Operation *, AffineMap> &opPermuationMap)
      : fusionStrategy(fusionStrategy), groupOpResults(groupOpResults),
        groupOpInitArgs(groupOpInitArgs), opPermuationMap(opPermuationMap) {}
  virtual ~CanonicalizerCommonUsedData() noexcept {};

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
        groupOpInitArgs.emplace_back(SetVector<Value>());
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

  void setPermutationMap(const DenseMap<Operation *, AffineMap> &map) {
    opPermuationMap = std::move(map);
  }

  // get methods
  VectorFusionStrategy &getFusionStrategy() noexcept { return fusionStrategy; }

  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8> &
  getGroupOpResults() noexcept {
    return groupOpResults;
  }

  SmallVector<SetVector<Value>, 8> &getGroupOpInitArgs() noexcept {
    return groupOpInitArgs;
  }

  DenseMap<Operation *, AffineMap> &getOpPermuationMap() noexcept {
    return opPermuationMap;
  }

  SmallVector<MultiReductionCanonicalizer, 8> &
  getMultiRdCanonicalizers() noexcept {
    return multiRdCanonicalizers;
  }

  llvm::SmallVector<BroadcastCanonicalizer, 8> &
  getBroadcastCanonicalizers() noexcept {
    return broadcastCanonicalizers;
  }

  llvm::SmallVector<TransposeCanonicalizer, 8> &
  getTransposeCanonicalizers() noexcept {
    return transposeCanonicalizers;
  }

  llvm::SmallVector<ShapeCastCanonicalizer, 8> &
  getShapeCastCanonicalizers() noexcept {
    return shapeCastCanonicalizers;
  }

  // other methods
  bool isGroupHasSpecialOperation(const size_t grpIdx);

  void generateEmptyTensorAndWrite(
      Operation *sourceOp,
      llvm::DenseMap<Operation *, std::pair<Value, Value>>
          &srcOpCanoniclizedMap,
      size_t anchorPos, ReturnTypeKind retKind,
      DenseMap<Operation *, size_t> &visitedOperation);

  void updateOpOperandResultInGroups(size_t opGid, Operation *op,
                                     const Value &init = Value(),
                                     const Value &result = Value());
  void removeOpInCurrentGroups(size_t grpIdx, Operation *op,
                               Operation *replacedOp);
  void updateOpGroupInfo(size_t grpIdx);

  Value
  canonicalizeCurrentOperation(Operation *op, const Value &transferReadOperand,
                               size_t operandIdx,
                               vector::TransferReadOp *srcReadOp = nullptr);

  Operation *
  createTransferReadOpBefore(Operation *op, const Value &operand,
                             vector::TransferReadOp *srcReadOp = nullptr);
  /// get next operation in current operation group
  template <class Target>
  Operation *getNextTargetOperationInCurrentGroup(Operation *curOp,
                                                  const size_t grpIdx);
};

/// generate for loop for each operation.
class ForLoopGenerator : virtual public CanonicalizerCommonUsedData {
private:
  func::FuncOp func;

public:
  ForLoopGenerator() = default;
  ForLoopGenerator(func::FuncOp &func) : func(func) {}

  virtual ~ForLoopGenerator() noexcept {}

  void setGeneratorFunc(func::FuncOp &func) noexcept { this->func = func; }
  void clearCurrentOperationGroup(size_t grpIdx);
  void generateGroupOpVectorizedIR(const int idx);

  /// prepare for loop iteration args
  void prepareForLoopArgs(const size_t grpIdx, GenerateLoopHelper &loopHelper);

  /// replace original operation result with corresponding for loop result
  void replaceOpUsersWithForLoopResult(
      scf::ForOp forOp, int grpIdx, SmallVector<Value, 4> &nextAnchorResults,
      DenseMap<Value, int> &nextAnchorResultsIdxMap,
      DenseMap<Value, Value> &forResultOrignalResultMap);

  /// mark which operation need to set correct for loop var idx
  /// due to sometimes we need to chage for loop order like reduce operation.
  void getCurrentGroupIndiceLoopMap(
      DenseMap<Operation *, DenseMap<size_t, size_t>> &indiceLoopMap,
      const size_t groupId, Operation *op,
      const DenseMap<size_t, size_t> &setIdxMap = DenseMap<size_t, size_t>({}));
  void
  rewriteOperationAsVectorize(OpBuilder &rewriter, size_t groupId,
                              const std::queue<Operation *> *queue = nullptr);
  void createNewConstantOp(Operation *srcOp,
                           vector::TransferWriteOp *transferWriteOp,
                           size_t groupSteps);
  // elementwise for loop
  mlir::FailureOr<scf::ForOp>
  generateVectorizedForLoop(const size_t groupId, IRRewriter &rewriter,
                            const VectorType vectorType);

  scf::ForOp constructNestedForOp(const size_t groupIdx, OpBuilder &b,
                                  const Location &loc, ArrayRef<int64_t> dims,
                                  GenerateLoopHelper &loopGenerator);

  void moveOperationsToCurrentForBody(
      const size_t groupIdx, const OpBuilder &b, ArrayRef<Value> inductionVars,
      const llvm::DenseMap<Value, int> &operandIdxMap, ValueRange loopState,
      DenseMap<Value, Value> &originalOperandLoopArgsMap,
      std::queue<Operation *> &queue,
      DenseMap<Operation *, DenseMap<size_t, size_t>> &indiceLoopMap);

  void setOperationCorrectOperand(
      Operation *op, ValueRange iterArgs,
      const DenseMap<Value, int> &operandIdxMap,
      DenseMap<Value, Value> &originalOperandLoopArgsMap,
      ArrayRef<Value> inductionVars,
      const DenseMap<Operation *, AffineMap> &opPermuationMap,
      DenseMap<Operation *, DenseMap<size_t, size_t>> &indiceLoopMap);

  void getResultInCurrentOps(const size_t anchorIdx, const size_t groupId,
                             const std::queue<Operation *> ops,
                             SmallVector<Value, 4> &results,
                             DenseMap<Value, int> &nextAnchorResultsIdxMap,
                             DenseMap<Value, Value> &forResultOrignalResultMap);

  /// todo: need to add a struct to remove so many parameters
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
                           DenseMap<Value, Value> &forResultOrignalResultMap,
                           ValueRange loopState,
                           DenseMap<Value, Value> &currentOperandOriginMap,
                           DenseMap<Value, int> &nextOperandIdxMap);

  /// todo: need to add a struct to remove so many parameters
  void movePostOpToCurrentAnchor(OpBuilder &b,
                                 GenerateLoopHelper &loopHelperParam);

  void movePreOpToCurrentAnchor(OpBuilder &b,
                                DenseMap<Value, int> &nextLoopStateIdxMap,
                                SmallVector<Value, 4> &nextAnchorArgs,
                                GenerateLoopHelper &loopHelperParam);

  void replaceOperationsWithForLoopResult(
      IRRewriter &rewrite, const ValueRange &forResults, const Block *forBlock,
      const llvm::SmallVector<Value, 4> &nextAnchorResults,
      const std::queue<Operation *> &movingOperations,
      DenseMap<Value, Value> &forResultOrignalResultMap);
  // multireduction forloop  methods
  scf::ForOp generateMultiReductionForLoop(const size_t grpIdx);
  /// Rearrange the current opIR to facilitate the generation of the correct
  /// reduction IR
  void rearrageMultiReductionIR(
      const size_t grpIdx,
      DenseMap<Operation *, DenseMap<size_t, size_t>> &indiceLoopMap);

  scf::ForOp reductionAxisGenerateForLoop(OpBuilder &opBuilder,
                                          const size_t reductionIdx,
                                          GenerateLoopHelper &loopHelperParam);

  scf::ForOp parallelAxisGenerateForLoop(
      OpBuilder &opBuilder,
      DenseMap<Operation *, DenseMap<size_t, size_t>> &indiceLoopMap,
      GenerateLoopHelper &loopHelperParam);

  vector::TransferReadOp cloneReductionTransferRead(
      Value &source, OpBuilder &b, IRMapping &readMap,
      const llvm::SmallVector<int64_t, 4> &parallelAxis,
      llvm::SmallVector<Value, 5> &inductionVars, bool lastDimReduction,
      MultiReduceOpAxisKind rdKind = MultiReduceOpAxisKind::Parallel);

  /// generate for loop for transpose operation
  scf::ForOp generateTransposeForLoop(const size_t groupId);
  scf::ForOp generateTransposeForLoopWithLastDim(
      OpBuilder &opBuilder, const size_t grpIdx, const size_t forDimIdx,
      const int tpSteps, const Location &loc, SmallVector<Value> &inductionVars,
      ValueRange iterArgs, DenseMap<Value, int> &operandIdxMap,
      DenseMap<Value, Value> &originalOperandMap, Operation *successorWriteOp);

  scf::ForOp generateTransposeScalarDataMovement(
      OpBuilder &opBuilder, const size_t grpIdx, const size_t forDimIdx,
      const Location &loc, SmallVector<Value> &inductionVars,
      const ValueRange &iterArgs, DenseMap<size_t, size_t> &tpAxisMap);

  // shapecast
  scf::ForOp generateShapeCastForLoop(const size_t grpIdx);
  scf::ForOp generateShapeCastReadWriteLoop(
      OpBuilder &opBuilder, const size_t grpIdx, const size_t forDimIdx,
      const size_t steps, const Location &loc,
      SmallVector<Value> &inductionVars, const ValueRange &iterArgs);
  /// rectify indice for transfer_write operation
  /// e.g.: vector.transfer_write"(%16, %9, %c0, %c0), the first %c0 should use
  /// original indice not create by us
  void rectifyWriteOperationIndice(vector::TransferWriteOp *originalWriteOp,
                                   SmallVectorImpl<Value> &writeVars);
  /// rectify indice for transfer_read operation, like broadcast operation
  /// fusion by transfer_read , but the transfer_read operation is in innermost
  /// for loop body, we must set correct for loop var. e.g.:
  /// vector.transfer_read"(%16, %9, %c0), the first %c0 should use correct for
  /// innermost loop iter vars
  void rectifyReadOperationIndice(vector::TransferReadOp *originalReadOp,
                                  VectorType loopType,
                                  ArrayRef<Value> inductionVars,
                                  SmallVectorImpl<Value> &readVars);

  /// rectify each group operand use  for loop result
  void rectifyGroupOperands(size_t currentGroupId, Value originalResult,
                            Value forResult);
};

class VectorOperationAnalyzer : virtual public CanonicalizerCommonUsedData {
private:
  func::FuncOp func;
  DenseMap<Operation *, std::pair<Value, Value>> srcOpCanoniclizedMap;
  DenseMap<Operation *, size_t> visitedOperation;

public:
  virtual ~VectorOperationAnalyzer() = default;
  VectorOperationAnalyzer() = default;
  VectorOperationAnalyzer(func::FuncOp &func) : func(func) {}

  void setAnalysisFunc(func::FuncOp &func) { this->func = func; }
  ///  remove the useless operation, due to it result is not require by other
  // operation
  void analysisEmptyGroup();
  /// get each operation in each group maximum support vectorization length
  void analysisGroupMaxSteps();
  /// analysis operation result of current group whether needed by other
  /// operation
  void analysisGroupOperaion();

  void specialOperationRectify(DenseMap<Operation *, size_t> &visitedOperation);
  /// update operation result kind
  void updateReturnResultKind(Operation *sourceOp, size_t sourceOpGid,
                              ReturnTypeKind rtKind);

  /// process the operation which need to return result
  /// \param *op current operation
  void groupOperationNeedReturnResult(size_t sourceOpGid, Operation *sourceOp,
                                      Operation *op, size_t operandIdx,
                                      bool inSameGroupNeedReturn);
  /// source operation write it's result to a tensor
  void makeSourceOpWriteResultToTensor(Operation *sourceOp, size_t sourceOpGid,
                                       ReturnTypeKind rtKind);
  /// analysis constant operation and replace it with a new constant operation
  void replaceConstantOpAsNewOp(Operation *op, Operation *sourceOp,
                                size_t operandIdx);
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
  virtual ~CanonicalizerVectorOperation() = default;

  // get functions
  func::FuncOp &getFunc() noexcept { return func; };
  IRRewriter &getIRWewriter() noexcept { return rewriter; }
  template <class T, class U>
  void processSpecialOperation(T &canonicalizers,
                               std::function<void(const size_t)> generateFunc);
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