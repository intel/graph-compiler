//===- TilingVector.hpp - Tiling large vector to small vector ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef GC_PASSES_TILINGVECTOR_H
#define GC_PASSES_TILINGVECTOR_H

#include "gc/Analysis//VectorBasedFusionAnalysis.h"
#include "gc/Analysis/TargetDescriptionAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir {
namespace gc {

/// get fusion kind
/// Has two kind:
/// 1. OperationGroup:
///     The operation is converted into physical registers through our fusion
///     strategy.
/// 2. Operations:(TODO:)
///     The user ensures that there is no data dependency between operations,
///     and we directly convert the operations into physical register sizes.
enum CanonicalizerKind { GroupOperations, Operations };

/// To avoid too many parameters in function when generate for loop
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

  /// update loop iteration args data
  void updateCurrentArgsStatus(DenseMap<Value, int> &currentArgsIdxMap,
                               SmallVector<Value, 4> &currentArgs,
                               DenseMap<Value, Value> &originalArgsMap,
                               DenseMap<Value, Value> &argsOriginalMap);
};

//===----------------------------------------------------------------------===//
// vectorize operation class
//===----------------------------------------------------------------------===//

/// base class of special operation
template <class T> class SpecialOperationCanonicalizer {
private:
  /// store current special operation
  SmallVector<T, 4> candidateRdOps;
  /// vectorize step
  size_t vectorStep = 1;

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
  SpecialOperationCanonicalizer(const SmallVector<T, 4> &candidateRdOps,
                                SpecialOperationKind kind)
      : candidateRdOps(candidateRdOps), kind(kind) {}
  SpecialOperationCanonicalizer(const SmallVector<T, 4> &candidateRdOps,
                                SpecialOperationKind kind, size_t step)
      : candidateRdOps(candidateRdOps), vectorStep(step), kind(kind) {}
  llvm::SmallVector<T, 4> &getCandidateOps();
  virtual ~SpecialOperationCanonicalizer() {}
  virtual void prepareSpecialOperationInfo() = 0;
  /// get kind of speical operation
  SpecialOperationKind getKind() noexcept { return kind; }
  /// set current operation group vectorize step
  void setVectorStep(size_t step) noexcept { vectorStep = step; }
  /// get current operation group vectorize step
  size_t getVectorStep() noexcept { return vectorStep; }
};

enum class MultiReduceOpAxisKind { Reduction, Parallel };
/// Help to vectorize reduction operation
class MultiReductionCanonicalizer
    : public SpecialOperationCanonicalizer<vector::MultiDimReductionOp> {
private:
  /// reduction parallel axis and reduction axis
  SmallVector<int64_t, 4> reductionAxis, parallelAxis;
  /// operations before reduction operation and operations after reduction
  /// operation
  std::queue<Operation *> prevOps, postOps, accRelatedOps, sourceRelatedOps;
  bool haslastDimReduction = false;
  bool isStandaloneOp = false;
  /// empty reduction means that all the reduction axis is 1
  bool isEmptyReduction = true;
  /// vector type rank
  int64_t typeRank = -1;
  /// record original operation result
  SetVector<Value> originalOpResults;
  /// vector type of source operation and accumulate operation
  VectorType sourceType, accType;
  /// for loop yield result index map
  llvm::SmallDenseMap<Value, int> resultIdxMap;

public:
  MultiReductionCanonicalizer(
      const SmallVector<vector::MultiDimReductionOp, 4> &candidateRdOps,
      size_t steps = 1)
      : SpecialOperationCanonicalizer<vector::MultiDimReductionOp>(
            candidateRdOps, SpecialOperationKind::OP_MultiDimReduction, steps) {
    isStandaloneOp = candidateRdOps.size() == 1;
  };
  virtual ~MultiReductionCanonicalizer() noexcept {};
  /// get reduction vector type, we use source operation type as reduction
  /// vector type
  int64_t getTypeRank();
  /// get reduction operation reduction and parallel axis
  void getReductionAxisAndParallelAxis();
  /// whether last dim is reduction axis
  bool hasLastDimReduction();
  /// whether only reduction operation in current operation group
  bool getIsStandAloneOp() noexcept { return isStandaloneOp; }
  /// get whether last dim is reduction axis
  bool getHasLastDimReduction() noexcept { return haslastDimReduction; }
  /// initialize to get reduction axis
  void initReductionAxis();
  /// initialize to get parallel axis
  void initParallelAxis();
  /// get reduction axis
  SmallVector<int64_t, 4> &getReductionAxis() noexcept {
    return reductionAxis;
  };
  /// get parallel axis
  SmallVector<int64_t, 4> &getParallelAxis() noexcept { return parallelAxis; };
  /// get prev operation in current operation group
  std::queue<Operation *> &getPrevOps() noexcept { return prevOps; }
  /// get post operation in current operation group
  std::queue<Operation *> &getPostOps() noexcept { return postOps; }
  /// get accumulate operation in reduction operation
  std::queue<Operation *> &getAccRelatedOps() noexcept { return accRelatedOps; }
  /// get source operation in reduction operation
  std::queue<Operation *> &getSourceRelatedOps() noexcept {
    return sourceRelatedOps;
  }
  /// get reduction operation original result
  SetVector<Value> &getOriginalOpResults() noexcept {
    return originalOpResults;
  }
  /// get source operation vector type
  VectorType getSourceType() noexcept { return sourceType; };
  /// get accumulate operation vector type
  VectorType getAccType() noexcept { return accType; };
  /// get result index map
  llvm::SmallDenseMap<Value, int> &getResultIdxMap() noexcept {
    return resultIdxMap;
  }
  /// set result index map
  void setResultIdxMap(const llvm::SmallDenseMap<Value, int> &map) {
    resultIdxMap = map;
  }

  /// initalize parallel, reduction axis, reduction operation type and whether
  /// last dim is reduction axis
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
      const SmallVector<vector::BroadcastOp, 4> &candidateBcOps,
      size_t steps = 1)
      : SpecialOperationCanonicalizer<vector::BroadcastOp>(
            candidateBcOps, SpecialOperationKind::OP_Broadcast, steps){};
  virtual ~BroadcastCanonicalizer() noexcept {}
  void prepareSpecialOperationInfo() override {}
  static bool classof(SpecialOperationCanonicalizer *canonicalizer) {
    return canonicalizer->getKind() == SpecialOperationKind::OP_Broadcast;
  }
};

class TransposeCanonicalizer
    : public SpecialOperationCanonicalizer<vector::TransposeOp> {
private:
  /// first and second transpose axis
  size_t firstTpIdx = 0, secondTpIdx = 0;

public:
  TransposeCanonicalizer(
      const llvm::SmallVector<vector::TransposeOp, 4> &candidateTpOps,
      size_t steps = 1)
      : SpecialOperationCanonicalizer<vector::TransposeOp>(
            candidateTpOps, SpecialOperationKind::OP_Transpose, steps){};
  virtual ~TransposeCanonicalizer() noexcept {}
  void prepareSpecialOperationInfo() override{};
  static bool classof(SpecialOperationCanonicalizer *canonicalizer) {
    return canonicalizer->getKind() == SpecialOperationKind::OP_Transpose;
  }
  enum TRANSPOSE_KERNEL {
    KERNEL_16X16 = 16,
  };
  /// get first transpose axis
  size_t getFirstTpIdx() noexcept { return firstTpIdx; }
  /// get second transpose axis
  size_t getSecondTpIdx() noexcept { return secondTpIdx; }
  /// whether transpose on two dimensions
  bool isTwoDTranspose();
  /// whether transpose on all dimension size is one
  bool isTransposeOnAllOneDim();
  /// whether transpose on last dimension
  bool transposeOnLastDim();
};

class ShapeCastCanonicalizer
    : public SpecialOperationCanonicalizer<vector::ShapeCastOp> {
private:
public:
  ShapeCastCanonicalizer(
      const SmallVector<vector::ShapeCastOp, 4> &candidateScOps,
      size_t steps = 1)
      : SpecialOperationCanonicalizer<vector::ShapeCastOp>(
            candidateScOps, SpecialOperationKind::OP_ShapeCast, steps){};
  virtual ~ShapeCastCanonicalizer() {}
  void prepareSpecialOperationInfo() override {}
  static bool classof(SpecialOperationCanonicalizer *canonicalizer) {
    return canonicalizer->getKind() == SpecialOperationKind::OP_ShapeCast;
  }
  /// whether store and load on last dimension
  bool isReadWriteOnLastDim();
};

/// generate for loop for each operation.
class ForLoopGenerator {
private:
  GroupOperationFusion vectorBasedFusion;

public:
  ForLoopGenerator(GroupOperationFusion &fusion) : vectorBasedFusion(fusion) {}

  virtual ~ForLoopGenerator() noexcept {}

  void setVectorBaseFusion(GroupOperationFusion &vectorBasedFusion) {
    this->vectorBasedFusion = vectorBasedFusion;
  };

  /// clear current group operation
  void clearCurrentOperationGroup(size_t grpIdx);

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

  // get methods
  GroupOperationFusion &getVectorBasedFusion() noexcept {
    return vectorBasedFusion;
  }
  /// rewrite operation as vectorize IR in current operation group
  void
  rewriteOperationAsVectorize(OpBuilder &rewriter, size_t groupId,
                              const std::queue<Operation *> *queue = nullptr);
  /// Reimplementation of writing a tensor from a constant of denseElementattr.
  void createNewConstantOp(Operation *srcOp,
                           vector::TransferWriteOp *transferWriteOp,
                           size_t groupSteps);
  // Generate elementwise operation for loop
  mlir::FailureOr<scf::ForOp>
  generateVectorizedForLoop(const size_t groupId, IRRewriter &rewriter,
                            const VectorType vectorType);
  scf::ForOp constructNestedForOp(const size_t groupIdx, OpBuilder &b,
                                  const Location &loc, ArrayRef<int64_t> dims,
                                  GenerateLoopHelper &loopHelper);
  /// move operations in \param queue to current loop anchor
  void moveOperationsToCurrentForBody(const OpBuilder &b,
                                      std::queue<Operation *> &queue,
                                      GenerateLoopHelper &loopHelperParam);

  /// Set correct operand with loop args for the operation
  void setOperationCorrectOperand(
      Operation *op, const DenseMap<Operation *, AffineMap> &opPermuationMap,
      GenerateLoopHelper &loopHelperParam);

  ///  Get current anchor return retults
  void getResultInCurrentOps(const size_t anchorIdx, const size_t groupId,
                             const std::queue<Operation *> &ops,
                             SmallVector<Value, 4> &results,
                             DenseMap<Value, int> &nextAnchorResultsIdxMap,
                             DenseMap<Value, Value> &forResultOrignalResultMap);
  /// Get next anchor's iteration loop args
  void getInitArgsToNextAnchor(llvm::DenseMap<Value, int> &nextAnchorArgsIdxMap,
                               llvm::SmallVector<Value, 4> &nextAnchorArgs,
                               GenerateLoopHelper &loopHelperParam);
  /// Get operation should appear in current loop anchor
  void getOperationInCurrentAnchor(const size_t anchorIdx,
                                   std::queue<Operation *> &fromQueue,
                                   std::queue<Operation *> &toQueue);
  /// Get current loop operation result
  void generateLoopResults(OpBuilder &b, const Location &loc,
                           GenerateLoopHelper &loopHelperParam,
                           DenseMap<Value, int> &nextOperandIdxMap);

  /// Move post operations in current operation group to the for loop body
  void movePostOpToCurrentAnchor(OpBuilder &b,
                                 GenerateLoopHelper &loopHelperParam);

  /// Move previous operations in current operation group to the for loop body
  void movePreOpToCurrentAnchor(OpBuilder &b,
                                DenseMap<Value, int> &nextLoopStateIdxMap,
                                SmallVector<Value, 4> &nextAnchorArgs,
                                GenerateLoopHelper &loopHelperParam);

  /// replace moved operation result used by current post operations with for
  /// loop result
  void replaceOperationsWithForLoopResult(
      IRRewriter &rewrite, const std::queue<Operation *> &movingOperations,
      GenerateLoopHelper &loopHelperParam);

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

class LoopGeneratorImpl : public ForLoopGenerator {

private:
  SmallVector<MultiReductionCanonicalizer, 8> multiRdCanonicalizers;
  SmallVector<BroadcastCanonicalizer, 8> broadcastCanonicalizers;
  SmallVector<TransposeCanonicalizer, 8> transposeCanonicalizers;
  SmallVector<ShapeCastCanonicalizer, 8> shapeCastCanonicalizers;

public:
  LoopGeneratorImpl(GroupOperationFusion &fusion) : ForLoopGenerator(fusion){};

  virtual ~LoopGeneratorImpl() noexcept {};

  SmallVector<MultiReductionCanonicalizer, 8> &
  getMultiRdCanonicalizers() noexcept {
    return multiRdCanonicalizers;
  }

  SmallVector<BroadcastCanonicalizer, 8> &
  getBroadcastCanonicalizers() noexcept {
    return broadcastCanonicalizers;
  }

  SmallVector<TransposeCanonicalizer, 8> &
  getTransposeCanonicalizers() noexcept {
    return transposeCanonicalizers;
  }

  SmallVector<ShapeCastCanonicalizer, 8> &
  getShapeCastCanonicalizers() noexcept {
    return shapeCastCanonicalizers;
  }
  /// clear special operation canonicalizer container
  void clearSpecialOperationCanonicalizers();

  /// add a dummy special canonicalizer
  void dummyInitSpecialOperation(size_t steps);

  /// initialize all the speical operation canonicalizer
  void initSpeicalOperationCanonicalizers();

  /// generate for loop for current special operation use \param generateFunc
  template <class T, class U>
  void processSpecialOperation(
      T &canonicalizers, const std::function<void(const size_t)> &generateFunc);
  // Canonicalize special operation
  void canonicalizeSpecialOperation();

  /// whether \param grpIdx operation group has special operation
  bool isGroupHasSpecialOperation(const size_t grpIdx);

  // multireduction forloop  methods
  scf::ForOp generateMultiReductionForLoop(const size_t grpIdx);

  /// reduction operation reduction axis for loop
  scf::ForOp reductionAxisGenerateForLoop(OpBuilder &opBuilder,
                                          const size_t reductionIdx,
                                          GenerateLoopHelper &loopHelperParam);
  /// reduction operation parallel axis for loop
  scf::ForOp parallelAxisGenerateForLoop(OpBuilder &opBuilder,
                                         GenerateLoopHelper &loopHelperParam);
  /// ensure accumulate operation appear in parallel loop, inorder to have
  /// correct reduce fusion
  void ensureAccInParallelLoop(GenerateLoopHelper &loopHelperParam,
                               ArrayRef<int64_t> parallelAxis,
                               Value multiReductionAcc,
                               DenseMap<Value, int> &nextAnchorArgsIdxMap,
                               SmallVector<Value, 4> &nextAnchorArgs);

  /// Rearrange the current opIR to facilitate the generation of the correct
  /// reduction IR
  void rearrageMultiReductionIR(
      const size_t grpIdx,
      DenseMap<Operation *, DenseMap<size_t, size_t>> &indiceLoopMap);

  /// generate for loop for transpose operation
  scf::ForOp generateTransposeForLoop(const size_t grpIdx);
  /// shuffle instruction optimize for transpose operation
  scf::ForOp generateTransposeForLoopWithLastDim(
      OpBuilder &opBuilder, const int tpSteps, const Location &loc,
      Operation *successorWriteOp, GenerateLoopHelper &loopHelperParam);

  /// generate transpose operation for loop of simple data movement
  scf::ForOp
  generateTransposeScalarDataMovement(OpBuilder &opBuilder, const Location &loc,
                                      DenseMap<size_t, size_t> &tpAxisMap,
                                      GenerateLoopHelper &loopHelperParam);

  /// generate shapecast operation for loop
  scf::ForOp generateShapeCastForLoop(const size_t grpIdx);
  /// generate simple data movement for loop
  scf::ForOp generateShapeCastReadWriteLoop(
      OpBuilder &opBuilder, const size_t grpIdx, const size_t forDimIdx,
      const size_t steps, const Location &loc,
      SmallVector<Value> &inductionVars, ValueRange iterArgs);

  /// vectorize operations in current operation group
  void generateGroupOpVectorizedIR(const int idx);
};

/// group operation fusion implementation class
class GroupOperationFusionImpl : public GroupOperationAnalysis {
private:
  /// In which tensor is the result of the source operation stored, and the
  /// result of transfer_write.
  DenseMap<Operation *, std::pair<Value, Value>> srcOpCanoniclizedMap;
  /// have visited operations
  DenseMap<Operation *, size_t> visitedOperation;

public:
  virtual ~GroupOperationFusionImpl() = default;
  GroupOperationFusionImpl(func::FuncOp &func, HardWareInfo &info)
      : GroupOperationAnalysis(func, info) {}

  /// Generate emtpy tensor and write operations for operations that need to
  /// return their results, and generate read operations for operations that
  /// need to read parameters from the block.
  void canonicalizeEachOperationGroup();

  void specialOperationRectify(DenseMap<Operation *, size_t> &visitedOperation);
  /// update operation result kind
  void updateReturnResultKind(Operation *sourceOp, size_t sourceOpGid,
                              ReturnTypeKind rtKind);

  /// process the operation which need to return result
  /// \param *op current operation
  void GroupOperationReturnResultProcess(size_t sourceOpGid,
                                         Operation *sourceOp, Operation *op,
                                         size_t operandIdx,
                                         bool inSameGroupNeedReturn);
  /// source operation write it's result to a tensor
  void makeSourceOpWriteResultToTensor(Operation *sourceOp, size_t sourceOpGid,
                                       ReturnTypeKind rtKind);
  /// analysis constant operation and replace it with a new constant operation
  void replaceConstantOpAsNewOp(Operation *op, Operation *sourceOp,
                                size_t operandIdx);
  /// replace \param op in \param grpIdx operation group with \param replacedOp
  void removeOpInCurrentGroups(size_t grpIdx, Operation *op,
                               Operation *replacedOp);
  /// update operation in grpIdx group related information
  void updateOpGroupInfo(size_t grpIdx);
  /// make a transfer_read operation and  read the producer operation result
  Value
  canonicalizeCurrentOperation(Operation *op, const Value &transferReadOperand,
                               size_t operandIdx,
                               vector::TransferReadOp *srcReadOp = nullptr);
  /// update \param opGid operation group
  void updateOpOperandResultInGroups(size_t opGid, Operation *op,
                                     const Value &init = Value(),
                                     const Value &result = Value());

  /// make emtpy tensor and write the operation result to the tensor
  void generateEmptyTensorAndWrite(
      Operation *sourceOp,
      llvm::DenseMap<Operation *, std::pair<Value, Value>>
          &srcOpCanoniclizedMap,
      size_t anchorPos, ReturnTypeKind retKind,
      DenseMap<Operation *, size_t> &visitedOperation);

  /// make a transfer_read operation
  Operation *
  createTransferReadOpBefore(Operation *op, const Value &operand,
                             vector::TransferReadOp *srcReadOp = nullptr);
};
/// Vectorize vector operation with target machines max simd length.
class VectorOperationCanonicalizer {
private:
  GroupOperationFusionImpl fusion;
  LoopGeneratorImpl loopGenerator;
  CanonicalizerKind kind;
  func::FuncOp func;
  IRRewriter rewriter;

public:
  VectorOperationCanonicalizer(
      func::FuncOp &func, HardWareInfo &info,
      CanonicalizerKind kind = CanonicalizerKind::GroupOperations)
      : fusion(func, info), loopGenerator(fusion.getGroupOperationFusion()),
        kind(kind), rewriter(func) {}
  virtual ~VectorOperationCanonicalizer() = default;
  /// run the vector canonicalizer for the IR
  void run();
};
} // namespace gc
} // namespace mlir
#endif