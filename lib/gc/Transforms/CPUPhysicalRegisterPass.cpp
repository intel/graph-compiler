//===- CPUPhysicalRegisterPass.cpp - tiling as physical vector --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "TilingVector.hpp"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CPUPHYSICALREGISTERPASS
#include "gc/Transforms/Passes.h.inc"
#define DEBUG_TYPE "lower-to-physical-register-pass"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define SAFE_EXPAND(X) X
#define LDBG(X) LLVM_DEBUG(DBGS() << SAFE_EXPAND(X) << "\n")

#define ARITH_CAST_OPERATIONS                                                  \
  arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::BitcastOp,             \
      arith::FPToSIOp, arith::FPToUIOp, arith::SIToFPOp, arith::UIToFPOp,      \
      arith::TruncFOp, arith::TruncIOp

/// TODO: remove it in the future
bool enableDebugPrinter = true;
bool disableSpecialOp = false;

void printQueue(const std::queue<Operation *> &opQueue) {
  auto tempQ(opQueue);
  while (!tempQ.empty()) {
    auto cur = tempQ.front();
    LDBG(*cur);
    tempQ.pop();
  }
}

void printGroupOps(SmallVector<std::queue<Operation *>, 8> &opGroups) {
  for (auto [idx, grp] : llvm::enumerate(opGroups)) {
    LDBG("group id: " << idx);
    if (grp.empty())
      continue;

    LDBG("__________________ group start_____________");
    printQueue(grp);
    LDBG("__________________ group end_______________");
  }
}

static inline bool isBroadcastOp(Operation *op) {
  return isa_and_nonnull<vector::BroadcastOp>(op);
}

static inline bool isProducerOp(Operation *op) {
  return isa<affine::AffineApplyOp>(op);
}

static inline bool isCandidateMoveOperations(Operation *op) {
  return isa<tensor::ExtractSliceOp, tensor::InsertSliceOp, tensor::EmptyOp>(
      op);
}

static inline bool isReadOrWriteOperation(Operation *op) {
  return isa<vector::TransferReadOp, vector::TransferWriteOp>(op);
}

/// Get the index position of the first element that is true
static size_t getFirstTrueIndex(ArrayRef<bool> ararys) {
  for (size_t i = 0; i < ararys.size(); i++)
    if (!ararys[i])
      return i;

  return -1;
}

static inline bool isSpecialOp(Operation *op) {
  return isa<vector::TransposeOp, vector::ReductionOp, vector::BroadcastOp,
             vector::ShapeCastOp, vector::MultiDimReductionOp, func::CallOp>(
      op);
}

/// whether operation is a not support operation
bool isNotSupportOperation(Operation *op) {
  return isa<vector::MaskOp, vector::ConstantMaskOp, vector::MaskedLoadOp,
             vector::MaskedStoreOp, vector::CreateMaskOp>(op);
}

/// whether the vector operation is operate on dynamic shape
bool hasDynamicShape(Operation *op) {
  if (failed(getOperationVectorType(op))) {
    return false;
  }
  auto isDynamicShapedType = [](Value x) {
    if (auto type = dyn_cast<ShapedType>(x.getType()))
      if (ShapedType::isDynamicShape(type.getShape()))
        return true;

    return false;
  };
  // Check operands data type.
  if (llvm::any_of(op->getOperands(), [&isDynamicShapedType](Value x) {
        return isDynamicShapedType(x);
      })) {
    return true;
  }

  // Check results data type.
  if (llvm::any_of(op->getResults(), [&isDynamicShapedType](OpResult x) {
        return isDynamicShapedType(x);
      })) {
    return true;
  }

  return false;
}

// TODO: Need to support these operations in the future
bool hasNotSupportOperation(func::FuncOp *func) {
  auto walkRes = func->walk([](Operation *op) {
    if (isNotSupportOperation(op)) {
      LDBG("Operation do not support yet : " << *op << "\n");
      return WalkResult::interrupt();
    }
    if (hasDynamicShape(op)) {
      LDBG("Operation has dynamic shape: " << *op << "\n");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return walkRes != WalkResult::advance();
}

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

/// get float or integer dense attribute
/// \param [in,out] attr
template <typename T>
void getConstantDenseAttr(TypedAttr &attr, VectorType type,
                          DenseElementsAttr denseAttr) {
  using APX = std::conditional_t<std::is_same_v<T, DenseFPElementsAttr>,
                                 APFloat, APInt>;
  attr = T::get(type, denseAttr.getSplatValue<APX>());
}

/// Create a new arith constant operation according to the dense element attr
FailureOr<Value> createArithSplatConstantOp(IRRewriter &rewriter,
                                            const Location &loc,
                                            DenseElementsAttr valueType,
                                            VectorType newOperandType) {
  if (not valueType.isSplat())
    return failure();

  TypedAttr attr;
  if (isa<FloatType>(newOperandType.getElementType()))
    getConstantDenseAttr<DenseFPElementsAttr>(attr, newOperandType, valueType);
  else
    getConstantDenseAttr<DenseIntElementsAttr>(attr, newOperandType, valueType);

  return rewriter.create<arith::ConstantOp>(loc, attr)->getResults()[0];
}

/// whether the operation result need to be returned
/// \param anchorIdx resuilt produce operation anchor position
/// \param retType resuilt return type
bool needReturnResult(std::pair<ReturnTypeKind, size_t> &retType,
                      size_t anchorIdx) {
  return retType.first != ReturnTypeKind::RT_InGroup or
         retType.second < anchorIdx;
}

// Since we rewrite transfer_read and transfer_write, the `permutationmap` must
// be changed.
void setOpVectorizationPermutationMap(Operation *op, OpBuilder &rewriter,
                                      const ShapedType &tensorType,
                                      const AffineMap &permutationMap) {
  auto dimExpr = permutationMap.getResults();
  auto lastDim = dyn_cast<AffineDimExpr>(dimExpr.back());
  if (not isa<AffineDimExpr>(lastDim))
    llvm_unreachable("Must be AffineDimExpr.");

  SmallVector<AffineExpr, 1> affineExprs(1, lastDim);
  auto destAffineMap = AffineMap::get(tensorType.getRank(), 0, affineExprs,
                                      rewriter.getContext());
  SmallVector<bool, 1> inBounds(1, true);
  if (isa<vector::TransferWriteOp>(op)) {
    auto transferWriteOp = cast<vector::TransferWriteOp>(op);
    transferWriteOp.setPermutationMap(destAffineMap);
    transferWriteOp.setInBoundsAttr(rewriter.getBoolArrayAttr(inBounds));
  } else if (isa<vector::TransferReadOp>(op)) {
    auto transferReadOp = cast<vector::TransferReadOp>(op);
    transferReadOp.setPermutationMap(destAffineMap);
    transferReadOp.setInBoundsAttr(rewriter.getBoolArrayAttr(inBounds));
  }
}

// scf.for yield helper function
scf::YieldOp maybeYieldValue(OpBuilder &b, Location loc, ValueRange value) {
  bool hasRetVal = !value.empty();
  if (hasRetVal)
    return b.create<scf::YieldOp>(loc, value);
  else
    return b.create<scf::YieldOp>(loc);
}

Operation *createTensorEmptyBefore(Operation *op) {

  auto rtType = cast<ShapedType>(op->getResultTypes()[0]);
  IRRewriter reWriter(op);
  Block *block = op->getBlock();

  reWriter.setInsertionPoint(block, block->getOperations().begin());

  SmallVector<int64_t> shapes;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < rtType.getRank(); i++) {
    shapes.push_back(rtType.getDimSize(i));
    if (rtType.isDynamicDim(i))
      dynDims.push_back(
          reWriter.create<tensor::DimOp>(op->getLoc(), op->getResult(0), i));
  }
  auto emtpyOp = reWriter.create<tensor::EmptyOp>(
      op->getLoc(), rtType.getShape(), rtType.getElementType(), dynDims);
  return emtpyOp;
}

/// get the tensor that operation should write into
Value getOperationResultTensor(
    Operation *op, DenseMap<Operation *, size_t> &visitedOperation) {
  OpResult result = op->getResults()[0];
  for (Operation *x : result.getUsers()) {
    if (not isa<vector::TransferWriteOp>(x))
      continue;

    Value sourceTensor = x->getOperands()[1];
    Operation *srcOp = sourceTensor.getDefiningOp();
    if (not visitedOperation.contains(srcOp))
      continue;

    size_t pos = visitedOperation[srcOp];
    if (pos > visitedOperation[op])
      continue;

    return sourceTensor;
  }
  LDBG("Result not write back to tensor.");

  return createTensorEmptyBefore(op)->getResults()[0];
}

Operation *createTransferWriteOpAfter(Operation *op, const Value &dest) {
  auto rtType = cast<ShapedType>(op->getResultTypes()[0]);
  int64_t rank = rtType.getRank();
  auto dstType = cast<ShapedType>(dest.getType());
  IRRewriter reWriter(op);

  auto zero = reWriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);

  reWriter.setInsertionPointAfter(op);
  SmallVector<bool> inBoundsVal(rank, true);

  SmallVector<int64_t> shapes;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < rtType.getRank(); i++) {
    shapes.push_back(rtType.getDimSize(i));
    if (rtType.isDynamicDim(i))
      dynDims.push_back(
          reWriter.create<tensor::DimOp>(op->getLoc(), op->getResult(0), i));
  }
  return reWriter.create<vector::TransferWriteOp>(
      op->getLoc(),
      /*vector=*/op->getResult(0),
      /*source=*/dest,
      /*indices=*/SmallVector<Value>(dstType.getRank(), zero),
      /*inBounds=*/inBoundsVal);
}

Operation *GroupOperationFusionImpl::createTransferReadOpBefore(
    Operation *op, const Value &operand, vector::TransferReadOp *srcReadOp) {
  auto operandType = cast<ShapedType>(operand.getType());

  IRRewriter rewriter(op);
  auto zero =
      rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
  auto padValue = rewriter.create<arith::ConstantOp>(
      rewriter.getUnknownLoc(),
      rewriter.getZeroAttr(operandType.getElementType()));

  if (srcReadOp) {
    auto resultType = cast<ShapedType>(srcReadOp->getType());
    SmallVector<bool> inBoundsVal(resultType.getRank(), true);
    auto srcReadOpAffineMap = srcReadOp->getPermutationMap();
    // result of read operation should be same as operand
    auto t = rewriter.create<vector::TransferReadOp>(
        op->getLoc(),
        /*vectorType=*/
        VectorType::get(resultType.getShape(), resultType.getElementType()),
        /*source=*/operand,
        /*indices=*/SmallVector<Value>(operandType.getRank(), zero),
        /**affinemap*/ srcReadOpAffineMap,
        /*inBounds=*/inBoundsVal);
    DenseMap<Operation *, AffineMap> &permutationMap =
        getGroupOperationFusion().getOpPermuationMap();
    permutationMap[t] = srcReadOpAffineMap;
    getGroupOperationFusion().getOpAnchorPos()[t] =
        t.getVectorType().getRank() - 1;

    return t;
  }
  SmallVector<bool> inBoundsVal(operandType.getRank(), true);
  auto t = rewriter.create<vector::TransferReadOp>(
      op->getLoc(),
      /*vectorType=*/
      VectorType::get(operandType.getShape(), operandType.getElementType()),
      /*source=*/operand,
      /*indices=*/SmallVector<Value>(operandType.getRank(), zero),
      /**affinemap*/ padValue,
      /*inBounds=*/inBoundsVal);
  DenseMap<Operation *, AffineMap> &permutationMap =
      getGroupOperationFusion().getOpPermuationMap();
  permutationMap[t] = t.getPermutationMap();
  getGroupOperationFusion().getOpAnchorPos()[t] =
      t.getVectorType().getRank() - 1;

  return t;
}

// canonicalizing operation as tensor empty and transfer write the operation
// result into the empty tensor
[[nodiscard]] std::pair<Value, Value>
canonicalizeSourceOperation(Operation *op,
                            DenseMap<Operation *, size_t> &visitedOperation) {
  auto resultTensor = getOperationResultTensor(op, visitedOperation);
  auto writeOp = createTransferWriteOpAfter(op, resultTensor);
  return std::make_pair(resultTensor, writeOp->getResults()[0]);
}

[[nodiscard]] Value GroupOperationFusionImpl::canonicalizeCurrentOperation(
    Operation *op, const Value &transferReadOperand, size_t operandIdx,
    vector::TransferReadOp *srcReadOp) {
  // transfer_read operation
  auto readOp = createTransferReadOpBefore(op, transferReadOperand, srcReadOp);
  op->setOperand(operandIdx, readOp->getResults()[0]);
  return readOp->getResults()[0];
}

// __________________________________
// Speical operations canonicalization
// __________________________________

//===----------------------------------------------------------------------===//
// MultiReduce Operation
//===----------------------------------------------------------------------===//

void getOpSourceOps(Operation *op, DenseSet<Operation *> &srcOps) {
  SmallVector<Value> srcOperands = op->getOperands();
  std::deque<Value> srcOperandsQueue(srcOperands.begin(), srcOperands.end());
  DenseSet<Operation *> visited;
  visited.insert(op);
  while (!srcOperandsQueue.empty()) {
    Value accOperand = srcOperandsQueue.front();
    srcOperandsQueue.pop_front();
    Operation *accOperandOp = accOperand.getDefiningOp();
    if (!accOperandOp or visited.count(accOperandOp))
      continue;

    visited.insert(accOperandOp);
    srcOps.insert(accOperandOp);
    auto accOperandOperands = accOperandOp->getOperands();
    srcOperandsQueue.insert(srcOperandsQueue.end(), accOperandOperands.begin(),
                            accOperandOperands.end());
  }
}

bool isSrcRelated(const DenseSet<Operation *> &srcOps, Operation *op) {
  return srcOps.count(op);
}

void getPrevOps(std::queue<Operation *> &prevOps,
                std::queue<Operation *> &opQueue, Operation *currentOp) {
  while (!opQueue.empty() && currentOp != opQueue.front()) {
    prevOps.push(opQueue.front());
    opQueue.pop();
  }
}

void getPostOps(std::queue<Operation *> &postOps,
                std::queue<Operation *> &opQueue, Operation *currentOp) {
  // pop multireduction op
  if (currentOp != opQueue.front())
    llvm_unreachable(
        "Current operation is not the front operation of the operation queue.");

  opQueue.pop();
  while (!opQueue.empty()) {
    postOps.push(opQueue.front());
    opQueue.pop();
  }
}

void getReductionInitAttr(vector::MultiDimReductionOp &multiReductionOp,
                          Attribute &initValueAttr) {
  auto vecType = multiReductionOp.getSourceVectorType();
  auto resultElementType = vecType.getElementType();
  if (isa<FloatType>(resultElementType))
    initValueAttr = FloatAttr::get(
        resultElementType,
        getInitValForReduce(multiReductionOp.getKind(), vecType));
  else
    initValueAttr = IntegerAttr::get(
        resultElementType,
        getInitValForReduce<int64_t>(multiReductionOp.getKind(), vecType));
}

/// get multi_reduction operation accumulate value source related operations
/// \param srcOp accumulate value source operation
void classifyAccRelatedOps(std::queue<Operation *> &accRelatedOps,
                           std::queue<Operation *> &sourceRelatedOps,
                           Operation *srcOp, std::queue<Operation *> &prevOps) {
  DenseSet<Operation *> srcOpsSet;
  getOpSourceOps(srcOp, srcOpsSet);
  while (!prevOps.empty()) {
    auto op = prevOps.front();
    prevOps.pop();
    if (isSrcRelated(srcOpsSet, op) or op == srcOp)
      accRelatedOps.push(op);
    else
      sourceRelatedOps.push(op);
  }
}

void ForLoopGenerator::moveOperationsToCurrentForBody(
    const OpBuilder &b, std::queue<Operation *> &opQueue,
    GenerateLoopHelper &loopHelperParam) {
  auto &opPermuationMap = getVectorBasedFusion().getOpPermuationMap();
  auto tmpQ(opQueue);
  while (!tmpQ.empty()) {
    auto x = tmpQ.front();
    tmpQ.pop();
    x->moveBefore(b.getBlock(), b.getBlock()->end());
    // check operation type to set correct operand
    setOperationCorrectOperand(x, opPermuationMap, loopHelperParam);
  }
}

void ForLoopGenerator::getResultInCurrentOps(
    const size_t anchorIdx, const size_t groupId,
    const std::queue<Operation *> &ops, SmallVector<Value, 4> &results,
    DenseMap<Value, int> &nextAnchorResultsIdxMap,
    DenseMap<Value, Value> &forResultOrignalResultMap) {
  auto tmpQ(ops);
  llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>> &groupResults =
      getVectorBasedFusion().getGroupOpResults()[groupId];
  while (!tmpQ.empty()) {
    Operation *cur = tmpQ.front();
    tmpQ.pop();
    auto curResult = cur->getResults()[0];
    if (groupResults.contains(curResult)) {
      std::pair<ReturnTypeKind, size_t> retType = groupResults[curResult];
      if (needReturnResult(retType, anchorIdx)) {
        results.emplace_back(curResult);
        nextAnchorResultsIdxMap[curResult] = results.size() - 1;
        forResultOrignalResultMap[curResult] = curResult;
      }
    }
  }
}

/// update loop args related status
/// \param nextAnchorArgsIdxMap anchor args index map
/// \param nextOriginalOperandMap original value to next loop args map
/// \param nextOperandOriginalMap next loop args to original value map
void updateCurrentArgsStatus(ValueRange loopState, const size_t loopStateIdx,
                             SmallVector<Value, 4> &nextAnchorArgs,
                             Value originalValue,
                             DenseMap<Value, int> &nextAnchorArgsIdxMap,
                             DenseMap<Value, Value> &nextOriginalOperandMap,
                             DenseMap<Value, Value> &nextOperandOriginalMap) {
  Value currentArgs = loopState[loopStateIdx];
  if (currentArgs.getType() != originalValue.getType()) {
    llvm::outs() << loopStateIdx << ","
                 << "\n";
    currentArgs.dump();
    llvm::llvm_unreachable_internal("Type not equal.");
  }
  nextAnchorArgs.emplace_back(currentArgs);
  nextAnchorArgsIdxMap[currentArgs] = nextAnchorArgs.size() - 1;
  nextOriginalOperandMap[originalValue] = currentArgs;
  nextOperandOriginalMap[currentArgs] = originalValue;
}

void ForLoopGenerator::getInitArgsToNextAnchor(
    DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs,
    GenerateLoopHelper &loopHelperParam) {
  DenseMap<Operation *, size_t> &opAnchorPos =
      getVectorBasedFusion().getOpAnchorPos();
  SetVector<Value> &opInitArgs =
      getVectorBasedFusion().getGroupOpInitArgs()[loopHelperParam.groupIdx];

  DenseSet<Value> visited;
  // find the next anchor arguments
  std::queue<Operation *> tmpQ(*loopHelperParam.candidateOps);
  DenseMap<Value, Value> nextOriginalOperandMap, nextOperandOriginalMap;

  while (!tmpQ.empty()) {
    Operation *cur = tmpQ.front();
    tmpQ.pop();
    auto curOperands = cur->getOperands();
    for (auto x : curOperands) {
      if (!visited.contains(x) and opInitArgs.contains(x) and
          opAnchorPos[cur] > loopHelperParam.anchorIdx) {
        if (not loopHelperParam.originalOperandLoopArgsMap.contains(x))
          llvm_unreachable("Must contains current value.");
        int loopStateIdx = loopHelperParam.currentLoopStateIdxMap
                               [loopHelperParam.originalOperandLoopArgsMap[x]];
        updateCurrentArgsStatus(loopHelperParam.loopIterArgs, loopStateIdx,
                                nextAnchorArgs, x, nextAnchorArgsIdxMap,
                                nextOriginalOperandMap, nextOperandOriginalMap);
        visited.insert(x);
      }
    }
  }
  loopHelperParam.originalOperandLoopArgsMap =
      std::move(nextOriginalOperandMap);
  loopHelperParam.loopArgsOriginalOperandMap =
      std::move(nextOperandOriginalMap);
}

void ForLoopGenerator::getOperationInCurrentAnchor(
    const size_t anchorIdx, std::queue<Operation *> &fromQueue,
    std::queue<Operation *> &toQueue) {
  while (!fromQueue.empty()) {
    Operation *curOp = fromQueue.front();
    if (anchorIdx == getVectorBasedFusion().getOpAnchorPos()[curOp]) {
      toQueue.push(curOp);
      fromQueue.pop();
      continue;
    }
    break;
  }
}

void ForLoopGenerator::replaceOperationsWithForLoopResult(
    IRRewriter &rewrite, const std::queue<Operation *> &movingOperations,
    GenerateLoopHelper &loopHelperParam) {
  auto tmpQ(movingOperations);
  DenseSet<Value> operationOperands;
  while (!tmpQ.empty()) {
    auto curOp = tmpQ.front();
    tmpQ.pop();
    for (auto x : curOp->getOperands())
      operationOperands.insert(x);
  }
  auto replaceIfFn = [&](OpOperand &use) {
    return operationOperands.contains(use.get());
  };
  for (auto [nxtForResult, nextLoopResult] :
       zip(loopHelperParam.forResults, loopHelperParam.nextAnchorResults)) {
    Value originalResult =
        loopHelperParam.nextAnchorResultOrignalResultMap[nextLoopResult];

    rewrite.replaceOpUsesWithIf(originalResult.getDefiningOp(), nxtForResult,
                                replaceIfFn);
  }
}

/// \param [in,out] nextLoopStateIdxMap
/// \param [in,out] nextAnchorArgs
void ForLoopGenerator::movePreOpToCurrentAnchor(
    OpBuilder &b, DenseMap<Value, int> &nextLoopStateIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs,
    GenerateLoopHelper &loopHelperParam) {

  // 1. get operations in current anchor position
  std::queue<Operation *> movingOperation;
  getOperationInCurrentAnchor(loopHelperParam.anchorIdx,
                              *loopHelperParam.candidateOps, movingOperation);

  // 2. rewrite operation as vectorize IR
  rewriteOperationAsVectorize(b, loopHelperParam.groupIdx, &movingOperation);

  // 3. move opeartions to current for block
  moveOperationsToCurrentForBody(b, movingOperation, loopHelperParam);

  // 4. get next anchor args
  getInitArgsToNextAnchor(nextLoopStateIdxMap, nextAnchorArgs, loopHelperParam);

  // 5. move operations to moved queue
  while (!movingOperation.empty()) {
    loopHelperParam.movedOps->push(movingOperation.front());
    movingOperation.pop();
  }
}

void ForLoopGenerator::movePostOpToCurrentAnchor(
    OpBuilder &b, GenerateLoopHelper &loopHelperParam) {

  std::queue<Operation *> movingOperations;
  // 1. get post-op to current loop bod
  getOperationInCurrentAnchor(loopHelperParam.anchorIdx,
                              *loopHelperParam.candidateOps, movingOperations);
  // 2. rewrite operation as vectorize IR
  rewriteOperationAsVectorize(b, loopHelperParam.groupIdx, &movingOperations);

  // 3. move opeartions to current for block
  moveOperationsToCurrentForBody(b, movingOperations, loopHelperParam);

  // 4. replace correct for loop result to post-op
  IRRewriter rewriter(b);
  replaceOperationsWithForLoopResult(rewriter, movingOperations,
                                     loopHelperParam);

  // 5. move operations to moved queue
  while (!movingOperations.empty()) {
    loopHelperParam.movedOps->push(movingOperations.front());
    movingOperations.pop();
  }
}

void ForLoopGenerator::generateLoopResults(
    OpBuilder &b, const Location &loc, GenerateLoopHelper &loopHelperParam,
    DenseMap<Value, int> &nextOperandIdxMap) {
  SmallVector<Value, 4> results;
  DenseMap<Value, Value> currentResultMap;
  getResultInCurrentOps(loopHelperParam.anchorIdx, loopHelperParam.groupIdx,
                        *loopHelperParam.movedOps, results,
                        loopHelperParam.nextAnchorResultsIdxMap,
                        currentResultMap);

  llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>> &groupResults =
      getVectorBasedFusion().getGroupOpResults()[loopHelperParam.groupIdx];
  // check for yield results whether need to return to next anchor
  for (auto [idx, forResult] :
       llvm::enumerate(loopHelperParam.nextAnchorResults)) {
    Value originalResult =
        loopHelperParam.nextAnchorResultOrignalResultMap[forResult];

    if (groupResults.contains(originalResult)) {
      std::pair<ReturnTypeKind, size_t> resultType =
          groupResults[originalResult];
      if (needReturnResult(resultType, loopHelperParam.anchorIdx)) {
        results.emplace_back(loopHelperParam.forResults[idx]);
        currentResultMap[loopHelperParam.forResults[idx]] = originalResult;
      }
    }
  }

  loopHelperParam.nextAnchorResults.clear();
  loopHelperParam.nextAnchorResultsIdxMap.clear();
  // reduction operation due to special process results size will be zero
  if (not results.empty())
    for (Value x : loopHelperParam.loopIterArgs) {
      loopHelperParam.nextAnchorResults.emplace_back(
          results[nextOperandIdxMap[x]]);
      loopHelperParam.nextAnchorResultsIdxMap[results[nextOperandIdxMap[x]]] =
          loopHelperParam.nextAnchorResults.size() - 1;
    }

  loopHelperParam.nextAnchorResultOrignalResultMap =
      std::move(currentResultMap);
}

void updateLoopArgsData(Value val, Value originalVal,
                        SmallVector<Value, 4> &argsArray,
                        DenseMap<Value, int> &anchorArgsIdxMap,
                        DenseMap<Value, Value> &originalOperandLoopArgsMap,
                        DenseMap<Value, Value> &loopArgsOriginalOperandMap) {
  argsArray.emplace_back(val);
  anchorArgsIdxMap[val] = argsArray.size() - 1;
  loopArgsOriginalOperandMap[val] = originalVal;
  originalOperandLoopArgsMap[originalVal] = val;
}

scf::ForOp LoopGeneratorImpl::reductionAxisGenerateForLoop(
    OpBuilder &opBuilder, const size_t reductionIdx,
    GenerateLoopHelper &loopHelperParam) {

  MultiReductionCanonicalizer rdCanonicalizer =
      getMultiRdCanonicalizers()[loopHelperParam.groupIdx];
  auto &multireductionOp = rdCanonicalizer.getCandidateOps()[0];
  GroupOperationFusion &fusionStrategy = getVectorBasedFusion();

  SmallVector<std::queue<Operation *>, 8> &opGroups =
      fusionStrategy.getOpGroups();
  std::queue<Operation *> &opQueue = opGroups[loopHelperParam.groupIdx];

  const auto loc = multireductionOp->getLoc();
  SmallVector<int64_t, 4> &reductionAxis = rdCanonicalizer.getReductionAxis();
  bool lastDimReduction = rdCanonicalizer.hasLastDimReduction();
  VectorType vectorType = rdCanonicalizer.getSourceType();
  const int loopStep =
      getVectorBasedFusion().getGroupMaxSteps()[loopHelperParam.groupIdx];
  func::FuncOp func = fusionStrategy.getFunction();
  IRRewriter rewriterOfFunc(func);

  Value zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  Value forSteps = makeIndexArithConstantOp(
      opBuilder, loc,
      (reductionIdx == reductionAxis.size() - 1 && lastDimReduction) ? loopStep
                                                                     : 1);
  Value numIter = makeIndexArithConstantOp(
      opBuilder, loc, vectorType.getShape()[reductionAxis[reductionIdx]]);
  scf::ForOp forOp = opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, loopHelperParam.loopIterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        loopHelperParam.inductionVars.emplace_back(iv);
        size_t currentAnchorId = loopHelperParam.anchorIdx;
        SmallVector<Value> tmpArgs(loopState);
        Value originalRetVal = multireductionOp->getResults()[0];

        if (reductionIdx < reductionAxis.size() - 1) {

          // 1. move pre-Op to current body
          DenseMap<Value, int> nextAnchorArgsIdxMap;
          SmallVector<Value, 4> nextAnchorArgs;
          std::queue<Operation *> movedOperation;
          DenseMap<Value, Value> currentoriginalArgsMap =
              loopHelperParam.originalOperandLoopArgsMap;
          DenseMap<Value, Value> currentArgsOriginalMap =
              loopHelperParam.loopArgsOriginalOperandMap;
          DenseMap<Value, int> currentArgsIdxMap =
              loopHelperParam.currentLoopStateIdxMap;
          DenseMap<Value, Value> originalArgsMap, argsOriginalMap;
          loopHelperParam.updateDataBeforePreOpMove(tmpArgs, opQueue,
                                                    movedOperation);
          movePreOpToCurrentAnchor(b, nextAnchorArgsIdxMap, nextAnchorArgs,
                                   loopHelperParam);
          loopHelperParam.updateDataAfterPreOpMove(nextAnchorArgsIdxMap,
                                                   nextAnchorArgs);

          // replace reduction init args
          if (currentoriginalArgsMap.contains(multireductionOp.getAcc())) {
            size_t accValIdx = currentArgsIdxMap
                [currentoriginalArgsMap[multireductionOp.getAcc()]];
            updateCurrentArgsStatus(
                loopState, accValIdx, nextAnchorArgs, multireductionOp.getAcc(),
                nextAnchorArgsIdxMap, originalArgsMap, argsOriginalMap);
            loopHelperParam.updateCurrentArgsStatus(
                nextAnchorArgsIdxMap, nextAnchorArgs, originalArgsMap,
                argsOriginalMap);
          }

          loopHelperParam.anchorIdx += 1;
          // 2. generate next for loop
          scf::ForOp nxtFor = reductionAxisGenerateForLoop(b, reductionIdx + 1,
                                                           loopHelperParam);
          loopHelperParam.anchorIdx -= 1;

          loopHelperParam.updateDataBeforePostOpMove(
              tmpArgs, currentArgsIdxMap, currentoriginalArgsMap,
              currentArgsOriginalMap, nxtFor->getResults(), b.getBlock(),
              movedOperation, currentAnchorId);
          // 3. move postOp to current body
          movePostOpToCurrentAnchor(b, loopHelperParam);

          // 4. generate loop results
          generateLoopResults(b, loc, loopHelperParam, nextAnchorArgsIdxMap);

          // reduction must return accumulate
          if (loopHelperParam.orignalResultNextAnchorResultMap.contains(
                  originalRetVal)) {
            Value lastForResult =
                loopHelperParam
                    .orignalResultNextAnchorResultMap[originalRetVal];
            size_t retIdx = nextAnchorArgsIdxMap
                [loopHelperParam
                     .nextAnchorResultOrignalResultMap[lastForResult]];
            Value forRes = nxtFor->getResults()[retIdx];
            // accumulate for loop iter args must be last, so we just put the
            // reduction result as the last result
            updateLoopArgsData(
                forRes, originalRetVal, loopHelperParam.nextAnchorResults,
                loopHelperParam.nextAnchorResultsIdxMap,
                loopHelperParam.orignalResultNextAnchorResultMap,
                loopHelperParam.nextAnchorResultOrignalResultMap);
          }

          maybeYieldValue(b, loc, loopHelperParam.nextAnchorResults);

        } else if (reductionIdx == reductionAxis.size() - 1) {
          std::queue<Operation *> movingOperation;

          while (!opQueue.empty()) {
            Operation *curOp = opQueue.front();
            opQueue.pop();
            if (isa<vector::MultiDimReductionOp>(curOp))
              break;

            movingOperation.push(curOp);
          }
          // remove all the multi_reduction operation
          while (!opQueue.empty()) {
            Operation *curOp = opQueue.front();
            if (isa<vector::MultiDimReductionOp>(curOp)) {
              opQueue.pop();
              continue;
            }
            break;
          }

          rewriteOperationAsVectorize(b, loopHelperParam.groupIdx,
                                      &movingOperation);
          loopHelperParam.loopIterArgs = loopState;
          moveOperationsToCurrentForBody(b, movingOperation, loopHelperParam);
          loopHelperParam.movedOps = &movingOperation;
          loopHelperParam.candidateOps = &opQueue;

          int accValIdx =
              loopHelperParam.currentLoopStateIdxMap
                  [loopHelperParam
                       .originalOperandLoopArgsMap[multireductionOp.getAcc()]];

          Value reductionResult = makeArithReduction(
              b, loc, multireductionOp.getKind(), multireductionOp.getSource(),
              loopState[accValIdx]);

          loopHelperParam.updateDataBeforePostOpMove(
              tmpArgs, loopHelperParam.currentLoopStateIdxMap,
              loopHelperParam.originalOperandLoopArgsMap,
              loopHelperParam.loopArgsOriginalOperandMap, ValueRange(),
              b.getBlock(), movingOperation, currentAnchorId);

          movePostOpToCurrentAnchor(b, loopHelperParam);

          loopHelperParam.nextAnchorResults.clear();
          updateLoopArgsData(reductionResult, originalRetVal,
                             loopHelperParam.nextAnchorResults,
                             loopHelperParam.nextAnchorResultsIdxMap,
                             loopHelperParam.orignalResultNextAnchorResultMap,
                             loopHelperParam.nextAnchorResultOrignalResultMap);
          getResultInCurrentOps(
              loopHelperParam.anchorIdx, loopHelperParam.groupIdx,
              movingOperation, loopHelperParam.nextAnchorResults,
              loopHelperParam.nextAnchorResultsIdxMap,
              loopHelperParam.nextAnchorResultOrignalResultMap);
          maybeYieldValue(b, loc, loopHelperParam.nextAnchorResults);
        }
      });

  return forOp;
}

void LoopGeneratorImpl::ensureAccInParallelLoop(
    GenerateLoopHelper &loopHelperParam, ArrayRef<int64_t> parallelAxis,
    Value multiReductionAcc, DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs) {
  if (loopHelperParam.anchorIdx == parallelAxis.size() - 1) {
    // Ensure accumalate expression appear in this parallel anchor
    // position. If it not appear in current anchor, we must move it in
    // here.
    //   1. delete it in operation queue
    //   2. move it in current movedqueue
    DenseSet<Value> argsSet(nextAnchorArgs.begin(), nextAnchorArgs.end());
    std::queue<Operation *> checkAccQueue(*loopHelperParam.movedOps);
    Value accInitVal;
    while (!checkAccQueue.empty()) {
      Operation *cur = checkAccQueue.front();
      checkAccQueue.pop();
      bool ok = false;
      for (auto x : cur->getResults()) {
        if (x == multiReductionAcc) {
          accInitVal = x;
          ok = true;
          break;
        }
      }
      if (ok)
        break;
    }
    if (accInitVal) {
      // we put initVal at last for loop args
      if (!argsSet.contains(accInitVal)) {
        nextAnchorArgs.emplace_back(accInitVal);
        nextAnchorArgsIdxMap[accInitVal] = nextAnchorArgs.size() - 1;
        loopHelperParam.loopArgsOriginalOperandMap[accInitVal] =
            multiReductionAcc;
        loopHelperParam.originalOperandLoopArgsMap[multiReductionAcc] =
            accInitVal;
      }
      loopHelperParam.loopIterArgs = nextAnchorArgs;
      loopHelperParam.nextAnchorResultsIdxMap = nextAnchorArgsIdxMap;
    } else {
      llvm::llvm_unreachable_internal("Wrong accumualte source value. Because "
                                      "acc value must appear in here.");
    }
  }
}

/// Generate for loop for parallel axis of `vector.multi_reduction`.
/// This function also call reduction axis for loop
scf::ForOp LoopGeneratorImpl::parallelAxisGenerateForLoop(
    OpBuilder &opBuilder, GenerateLoopHelper &loopHelperParam) {
  MultiReductionCanonicalizer &rdCanonicalizer =
      getMultiRdCanonicalizers()[loopHelperParam.groupIdx];
  vector::MultiDimReductionOp &multiReductionOp =
      rdCanonicalizer.getCandidateOps()[0];
  VectorType vectorType = rdCanonicalizer.getSourceType();
  GroupOperationFusion &fusionStrategy = getVectorBasedFusion();
  func::FuncOp func = fusionStrategy.getFunction();
  IRRewriter rewriterOfFunc(func);

  SmallVector<int64_t, 4> &parallelAxis = rdCanonicalizer.getParallelAxis();
  const Location &loc = multiReductionOp.getLoc();
  Value zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  size_t grpMaxStep =
      getVectorBasedFusion().getGroupMaxSteps()[loopHelperParam.groupIdx];
  size_t actualStep =
      (loopHelperParam.anchorIdx == parallelAxis.size() - 1 ? grpMaxStep : 1);
  Value forSteps = makeIndexArithConstantOp(opBuilder, loc, actualStep);

  // last dim reduction need to a generate dim=16 loop for fused with pre-op
  int dimSize = 0;
  if (loopHelperParam.anchorIdx == parallelAxis.size())
    dimSize =
        getVectorBasedFusion().getGroupMaxSteps()[loopHelperParam.groupIdx];
  else
    dimSize = vectorType.getShape()[parallelAxis[loopHelperParam.anchorIdx]];

  Value numIter = makeIndexArithConstantOp(opBuilder, loc, dimSize);
  // Create a loop and move vectorized operation into loops.
  return opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, loopHelperParam.loopIterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        loopHelperParam.inductionVars.emplace_back(iv);

        DenseMap<Operation *, size_t> &opIndexMap =
            fusionStrategy.getOpGroupIndexMap();

        if (not opIndexMap.contains(multiReductionOp))
          llvm_unreachable("Must constains multireduction operation.");

        size_t opIndex = opIndexMap[multiReductionOp];
        SmallVector<std::queue<Operation *>, 8> &opGroups =
            fusionStrategy.getOpGroups();
        std::queue<Operation *> &opQueue = opGroups[opIndex];
        Value multiReductionAcc = multiReductionOp.getAcc();

        if (loopHelperParam.anchorIdx < parallelAxis.size()) {
          // 1. move pre-Op to current body
          DenseMap<Value, int> nextAnchorArgsIdxMap;
          SmallVector<Value, 4> nextAnchorArgs;
          std::queue<Operation *> movedQueue;
          DenseMap<Value, Value> currentOriginalOperandMap =
              loopHelperParam.originalOperandLoopArgsMap;
          DenseMap<Value, Value> currentOperandOriginalMap =
              loopHelperParam.loopArgsOriginalOperandMap;
          DenseMap<Value, int> currentLoopStateIdxMap =
              loopHelperParam.currentLoopStateIdxMap;
          SmallVector<Value> tmpArgs(loopState);
          loopHelperParam.updateDataBeforePreOpMove(tmpArgs, opQueue,
                                                    movedQueue);
          movePreOpToCurrentAnchor(b, nextAnchorArgsIdxMap, nextAnchorArgs,
                                   loopHelperParam);
          loopHelperParam.updateDataAfterPreOpMove(nextAnchorArgsIdxMap,
                                                   nextAnchorArgs);
          ensureAccInParallelLoop(loopHelperParam, parallelAxis,
                                  multiReductionAcc, nextAnchorArgsIdxMap,
                                  nextAnchorArgs);
          scf::ForOp nxtFor;
          // 2. generate next for loop
          bool useParallelLoop =
              rdCanonicalizer.hasLastDimReduction() or
              loopHelperParam.anchorIdx < parallelAxis.size() - 1;
          loopHelperParam.anchorIdx += 1;
          if (useParallelLoop) {
            nxtFor = parallelAxisGenerateForLoop(b, loopHelperParam);
          } else {
            nxtFor = reductionAxisGenerateForLoop(b, 0, loopHelperParam);
          }
          loopHelperParam.anchorIdx -= 1;

          // 3. move postOp to current body
          loopHelperParam.updateDataBeforePostOpMove(
              tmpArgs, currentLoopStateIdxMap, currentOriginalOperandMap,
              currentOperandOriginalMap, nxtFor->getResults(),
              nxtFor->getBlock(), movedQueue, loopHelperParam.anchorIdx);
          movePostOpToCurrentAnchor(b, loopHelperParam);
          // 4. generate loop results
          generateLoopResults(b, loc, loopHelperParam, nextAnchorArgsIdxMap);
          maybeYieldValue(b, loc, loopHelperParam.nextAnchorResults);

        } else if (loopHelperParam.anchorIdx == parallelAxis.size()) {

          DenseMap<Value, Value> tmpOriginOperandLoopArgsMap =
              loopHelperParam.originalOperandLoopArgsMap;
          DenseMap<Value, Value> tmpLoopArgsOriginalOperandMap =
              loopHelperParam.loopArgsOriginalOperandMap;

          // get accumualte value
          Attribute initValueAttr;
          getReductionInitAttr(multiReductionOp, initValueAttr);

          auto accVal = b.create<arith::ConstantOp>(
              loc, DenseElementsAttr::get(
                       fusionStrategy.getTypeHelper().getVectorzedType(
                           multiReductionOp, dimSize),
                       {initValueAttr}));

          // put accumulte val at first for loop args
          DenseMap<Value, int> localAnchorArgsIdxMap;
          DenseMap<Value, Value> localOriginalOperandLoopArgsMap,
              localLoopArgsOriginalOperandMap;
          SmallVector<Value, 4> argsArray;
          updateLoopArgsData(
              accVal, multiReductionAcc, argsArray, localAnchorArgsIdxMap,
              localOriginalOperandLoopArgsMap, localLoopArgsOriginalOperandMap);

          size_t accLoopStateIdx =
              loopHelperParam.currentLoopStateIdxMap
                  [loopHelperParam
                       .originalOperandLoopArgsMap[multiReductionAcc]];
          for (auto [idx, x] : llvm::enumerate(loopState)) {
            if (idx == accLoopStateIdx)
              continue;
            updateLoopArgsData(x,
                               loopHelperParam.loopArgsOriginalOperandMap
                                   [loopHelperParam.loopIterArgs[idx]],
                               argsArray, localAnchorArgsIdxMap,
                               localOriginalOperandLoopArgsMap,
                               localLoopArgsOriginalOperandMap);
          }
          loopHelperParam.updateCurrentArgsStatus(
              localAnchorArgsIdxMap, argsArray, localOriginalOperandLoopArgsMap,
              localLoopArgsOriginalOperandMap);
          DenseMap<Value, Value> originalResultForResultMap;
          auto nxtFor = reductionAxisGenerateForLoop(b, 0, loopHelperParam);

          // insert accumulate value to original vector
          Value nxtForAccVal =
              originalResultForResultMap[multiReductionOp->getResults()[0]];
          size_t accIdx = loopHelperParam.nextAnchorResultsIdxMap[nxtForAccVal];
          auto accRes = nxtFor->getResults()[accIdx];

          Operation *reductionOp = b.create<vector::ReductionOp>(
              loc, multiReductionOp.getKind(), accRes);
          auto insertOp = b.create<vector::InsertOp>(
              loc, reductionOp->getResult(0), loopState[accLoopStateIdx], iv);

          // generate loop result
          SmallVector<Value> currentAnchorResults(loopState.size());
          DenseMap<Value, Value> currentResultMap;
          DenseMap<Value, int> currentResultIdxMap;

          currentAnchorResults[accLoopStateIdx] = insertOp->getResults()[0];
          // reduce axis for loop first result we has already processed above
          currentResultMap[insertOp->getResults()[0]] =
              multiReductionOp->getResults()[0];
          currentResultIdxMap[insertOp->getResults()[0]] = accLoopStateIdx;
          for (auto [idx, x] :
               llvm::enumerate(loopHelperParam.nextAnchorResults)) {
            if (loopHelperParam.nextAnchorResultOrignalResultMap[x] ==
                multiReductionOp->getResults()[0])
              continue;

            Value originalResult =
                loopHelperParam.nextAnchorResultOrignalResultMap[x];
            size_t itrIdx = loopHelperParam.currentLoopStateIdxMap
                                [tmpOriginOperandLoopArgsMap[originalResult]];
            currentAnchorResults[itrIdx] = nxtFor->getResults()[idx];
            currentResultIdxMap[nxtFor->getResults()[idx]] = itrIdx;
            currentResultMap[nxtFor->getResults()[idx]] = originalResult;
          }
          loopHelperParam.clearNextAnchorResults();
          loopHelperParam.setNextAnchorResults(
              currentAnchorResults, currentResultMap, currentResultIdxMap);
          maybeYieldValue(b, loc, loopHelperParam.nextAnchorResults);
        }
      });
}

scf::ForOp LoopGeneratorImpl::generateTransposeForLoopWithLastDim(
    OpBuilder &opBuilder, const int tpSteps, const Location &loc,
    Operation *successorWriteOp, GenerateLoopHelper &loopHelperParam) {
  auto &tpCanonicalizer =
      getTransposeCanonicalizers()[loopHelperParam.groupIdx];
  vector::TransposeOp &tpOp = tpCanonicalizer.getCandidateOps()[0];
  VectorType vtType = tpOp.getVector().getType();
  size_t rank = vtType.getRank();

  auto zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  bool isTransposeDim =
      loopHelperParam.anchorIdx == tpCanonicalizer.getFirstTpIdx() or
      loopHelperParam.anchorIdx == tpCanonicalizer.getSecondTpIdx();
  auto forSteps =
      makeIndexArithConstantOp(opBuilder, loc, isTransposeDim ? tpSteps : 1);
  auto numIter = makeIndexArithConstantOp(
      opBuilder, loc, vtType.getShape()[loopHelperParam.anchorIdx]);
  VectorType kernelType =
      VectorType::get({tpSteps, tpSteps}, vtType.getElementType());
  // generate transpose for loop
  return opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, loopHelperParam.loopIterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        loopHelperParam.inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (loopHelperParam.anchorIdx == rank - 1) {
          // transfer read from source tensor
          Value source = tpOp->getOperand(0);
          auto readSourceOp =
              cast<vector::TransferReadOp>(source.getDefiningOp());
          auto padValue = b.create<arith::ConstantOp>(
              loc, b.getZeroAttr(vtType.getElementType()));
          SmallVector<bool> inBoundsVal(2, true);
          inBoundsVal[0] = !ShapedType::isDynamic(
              vtType.getShape()[tpCanonicalizer.getFirstTpIdx()]);
          inBoundsVal[1] = !ShapedType::isDynamic(
              vtType.getShape()[tpCanonicalizer.getSecondTpIdx()]);

          auto transferReadOp = b.create<vector::TransferReadOp>(
              loc,
              /*vectorType=*/kernelType,
              /*source=*/readSourceOp.getSource(),
              /*indices=*/loopHelperParam.inductionVars,
              /*padding=*/padValue,
              /*inBounds=*/inBoundsVal);
          SmallVector<int64_t> perm{1, 0};
          auto transposeOp = b.create<vector::TransposeOp>(
              loc, transferReadOp->getResults()[0], perm);
          SmallVector<Value> writeVars(loopHelperParam.inductionVars.begin(),
                                       loopHelperParam.inductionVars.end());
          writeVars[tpCanonicalizer.getSecondTpIdx()] =
              loopHelperParam.inductionVars[tpCanonicalizer.getFirstTpIdx()];
          writeVars[tpCanonicalizer.getFirstTpIdx()] =
              loopHelperParam.inductionVars[tpCanonicalizer.getSecondTpIdx()];
          auto writeOp = b.create<vector::TransferWriteOp>(
              loc, transposeOp->getResults()[0], loopState[0], writeVars,
              inBoundsVal);
          maybeYieldValue(b, loc, writeOp->getResults());
        } else {
          // outter loop
          loopHelperParam.anchorIdx += 1;
          loopHelperParam.loopIterArgs = loopState;
          auto nxtFor = generateTransposeForLoopWithLastDim(
              b, tpSteps, loc, successorWriteOp, loopHelperParam);
          loopHelperParam.anchorIdx -= 1;
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
}

void ForLoopGenerator::prepareForLoopArgs(const size_t grpIdx,
                                          GenerateLoopHelper &loopHelper) {
  SetVector<Value> &grpArgs =
      getVectorBasedFusion().getGroupOpInitArgs()[grpIdx];
  loopHelper.loopIterArgs = grpArgs.getArrayRef();
  for (auto [idx, val] : llvm::enumerate(grpArgs)) {
    loopHelper.currentLoopStateIdxMap[val] = idx;
    loopHelper.originalOperandLoopArgsMap[val] = val;
    loopHelper.loopArgsOriginalOperandMap[val] = val;
  }
}

void LoopGeneratorImpl::rearrageMultiReductionIR(
    const size_t grpIdx,
    DenseMap<Operation *, DenseMap<size_t, size_t>> &indiceLoopMap) {
  MultiReductionCanonicalizer &rdCanonicalizer =
      getMultiRdCanonicalizers()[grpIdx];
  vector::MultiDimReductionOp multiReductionOp =
      rdCanonicalizer.getCandidateOps()[0];
  SmallVector<int64_t, 4> &parallelAxis = rdCanonicalizer.getParallelAxis();
  SmallVector<int64_t, 4> &reductionAxis = rdCanonicalizer.getReductionAxis();
  std::queue<Operation *> &prevOps = rdCanonicalizer.getPrevOps();
  std::queue<Operation *> &postOps = rdCanonicalizer.getPostOps();
  std::queue<Operation *> &accRelatedOps = rdCanonicalizer.getAccRelatedOps();
  std::queue<Operation *> &sourceRelatedOps =
      rdCanonicalizer.getSourceRelatedOps();
  std::queue<Operation *> &opQueue =
      getVectorBasedFusion().getOpGroups()[grpIdx];
  auto copyOpQueue(opQueue);
  getPrevOps(prevOps, copyOpQueue, multiReductionOp);
  getPostOps(postOps, copyOpQueue, multiReductionOp);
  classifyAccRelatedOps(accRelatedOps, sourceRelatedOps,
                        multiReductionOp.getAcc().getDefiningOp(), prevOps);

  // mark source read operation need to set correct for loop var idx
  std::queue<Operation *> tmpSourceQ(sourceRelatedOps);
  DenseMap<size_t, size_t> varLoopIdxMap;
  VectorType groupVector =
      getVectorBasedFusion().getGroupBiggestRankVectorType()[grpIdx];
  for (size_t i = 0; i < parallelAxis.size(); i++) {
    varLoopIdxMap[parallelAxis[i]] = i;
  }
  size_t offset = rdCanonicalizer.hasLastDimReduction() ? 1 : 0;
  for (size_t i = parallelAxis.size() + offset;
       i < groupVector.getRank() + offset; i++) {
    varLoopIdxMap[reductionAxis[i - parallelAxis.size() - offset]] = i;
  }
  while (!tmpSourceQ.empty()) {
    auto *curOp = tmpSourceQ.front();
    tmpSourceQ.pop();
    if (isa<vector::TransferReadOp>(curOp))
      getCurrentGroupIndiceLoopMap(indiceLoopMap, grpIdx, curOp, varLoopIdxMap);
  }

  // move accumulate related operation to operation first
  std::queue<Operation *> rectifyQueue;
  DenseSet<Operation *> pushedSet;
  auto moveOperation = [&](std::queue<Operation *> &from,
                           std::queue<Operation *> &to) {
    while (!from.empty()) {
      auto cur = from.front();
      from.pop();
      if (pushedSet.contains(cur))
        continue;

      to.push(cur);
      pushedSet.insert(cur);
    }
  };
  moveOperation(accRelatedOps, rectifyQueue);
  moveOperation(opQueue, rectifyQueue);
  opQueue = rectifyQueue;
}

void ForLoopGenerator::replaceOpUsersWithForLoopResult(
    scf::ForOp forOp, int grpIdx, SmallVector<Value, 4> &nextAnchorResults,
    DenseMap<Value, int> &nextAnchorResultsIdxMap,
    DenseMap<Value, Value> &forResultOrignalResultMap) {
  IRRewriter rewriter(forOp);
  DenseSet<Operation *> forOpChildOps;
  forOp->walk([&](Operation *op) { forOpChildOps.insert(op); });
  auto replaceIfFn = [&](OpOperand &use) {
    return not forOpChildOps.contains(use.getOwner());
  };
  for (auto x : nextAnchorResults) {
    auto originalResult = forResultOrignalResultMap[x];
    Value forResult = forOp->getResults()[nextAnchorResultsIdxMap[x]];
    // subsequent group must use the replaced result as operand
    rectifyGroupOperands(grpIdx, originalResult, forResult);
    rewriter.replaceOpUsesWithIf(originalResult.getDefiningOp(), forResult,
                                 replaceIfFn);
  }
}
scf::ForOp
LoopGeneratorImpl::generateMultiReductionForLoop(const size_t grpIdx) {

  DenseMap<Operation *, DenseMap<size_t, size_t>> indiceLoopMap;
  rearrageMultiReductionIR(grpIdx, indiceLoopMap);
  // get current loop init args
  DenseMap<Value, int> currentLoopStateIdxMap, nextAnchorResultsIdxMap;
  GenerateLoopHelper loopHelper(grpIdx, 0);
  prepareForLoopArgs(grpIdx, loopHelper);

  MultiReductionCanonicalizer &rdCanonicalizer =
      getMultiRdCanonicalizers()[grpIdx];

  OpBuilder opBuilder(rdCanonicalizer.getCandidateOps()[0]);
  loopHelper.indiceLoopMap = indiceLoopMap;

  scf::ForOp forOp = parallelAxisGenerateForLoop(opBuilder, loopHelper);
  replaceOpUsersWithForLoopResult(forOp, grpIdx, loopHelper.nextAnchorResults,
                                  loopHelper.nextAnchorResultsIdxMap,
                                  loopHelper.nextAnchorResultOrignalResultMap);

  vector::MultiDimReductionOp multiReductionOp =
      rdCanonicalizer.getCandidateOps()[0];
  IRRewriter rewriter(multiReductionOp);
  rewriter.eraseOp(multiReductionOp);

  return forOp;
}

// generate simple data movement for loop
scf::ForOp LoopGeneratorImpl::generateTransposeScalarDataMovement(
    OpBuilder &opBuilder, const Location &loc,
    DenseMap<size_t, size_t> &tpAxisMap, GenerateLoopHelper &loopHelperParam) {
  auto &tpCanonicalizer =
      getTransposeCanonicalizers()[loopHelperParam.groupIdx];
  vector::TransposeOp &tpOp = tpCanonicalizer.getCandidateOps()[0];
  VectorType vtType = tpOp.getSourceVectorType();
  size_t rank = vtType.getRank();

  auto zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  size_t vecStep = tpCanonicalizer.transposeOnLastDim()
                       ? tpCanonicalizer.getVectorStep()
                       : 1;
  auto forSteps = makeIndexArithConstantOp(
      opBuilder, loc, loopHelperParam.anchorIdx == rank - 1 ? (vecStep) : 1);
  auto numIter = makeIndexArithConstantOp(
      opBuilder, loc, vtType.getShape()[loopHelperParam.anchorIdx]);

  SmallVector<int64_t> vecShapes(1, vecStep);
  VectorType kernelType = VectorType::get(vecShapes, vtType.getElementType());
  // generate transpose for loop
  return opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, loopHelperParam.loopIterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        loopHelperParam.inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (loopHelperParam.anchorIdx == rank - 1) {
          // transfer read from source tensor
          Value source = tpOp->getOperand(0);
          auto readSourceOp =
              cast<vector::TransferReadOp>(source.getDefiningOp());
          vector::TransferWriteOp successorWriteOp;
          for (Operation *x : tpOp->getUsers()) {
            if (isa<vector::TransferWriteOp>(x)) {
              successorWriteOp = cast<vector::TransferWriteOp>(x);
              break;
            }
          }
          auto padValue = b.create<arith::ConstantOp>(
              loc, b.getZeroAttr(vtType.getElementType()));
          SmallVector<bool> inBoundsVal(1, true);
          SmallVector<Value> writeVars;
          size_t itrIdx = 0;
          while (itrIdx < rank) {
            writeVars.emplace_back(
                loopHelperParam.inductionVars[tpAxisMap[itrIdx]]);
            itrIdx++;
          }
          auto transferReadOp = b.create<vector::TransferReadOp>(
              loc,
              /*vectorType=*/kernelType,
              /*source=*/readSourceOp.getSource(),
              /*indices=*/loopHelperParam.inductionVars,
              /*padding=*/padValue,
              /*inBounds=*/inBoundsVal);

          rectifyWriteOperationIndice(&successorWriteOp, writeVars);

          auto writeOp = b.create<vector::TransferWriteOp>(
              loc, transferReadOp->getResults()[0], loopState[0], writeVars,
              inBoundsVal);
          maybeYieldValue(b, loc, writeOp->getResults());
        } else {
          // outter loop
          loopHelperParam.anchorIdx += 1;
          loopHelperParam.loopIterArgs = loopState;
          auto nxtFor = generateTransposeScalarDataMovement(b, loc, tpAxisMap,
                                                            loopHelperParam);
          loopHelperParam.anchorIdx -= 1;
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
}

scf::ForOp LoopGeneratorImpl::generateShapeCastReadWriteLoop(
    OpBuilder &opBuilder, const size_t grpIdx, const size_t forDimIdx,
    const size_t steps, const Location &loc, SmallVector<Value> &inductionVars,
    ValueRange iterArgs) {
  auto &scCanonicalizer = getShapeCastCanonicalizers()[grpIdx];
  vector::ShapeCastOp &scOp = scCanonicalizer.getCandidateOps()[0];
  VectorType sourceType = scOp.getSourceVectorType();
  VectorType destType = scOp.getResultVectorType();
  VectorType loopType =
      sourceType.getRank() > destType.getRank() ? sourceType : destType;
  size_t rank = loopType.getRank();
  DenseMap<Operation *, size_t> &opIndexMap =
      getVectorBasedFusion().getOpGroupIndexMap();

  auto zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  bool isLastDim = loopType.getRank() - 1 == (int64_t)forDimIdx;
  auto forSteps =
      makeIndexArithConstantOp(opBuilder, loc, isLastDim ? steps : 1);
  auto numIter =
      makeIndexArithConstantOp(opBuilder, loc, loopType.getShape()[forDimIdx]);
  VectorType kernelType =
      VectorType::get({(int64_t)steps}, loopType.getElementType());

  // generate transpose for loop
  return opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, iterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (forDimIdx == rank - 1) {
          // transfer read from source tensor
          Value source = scOp->getOperand(0);
          auto readSourceOp =
              cast<vector::TransferReadOp>(source.getDefiningOp());
          SmallVector<vector::TransferWriteOp> successorWriteOps;
          for (Operation *x : scOp->getUsers()) {
            if (isa<vector::TransferWriteOp>(x) and opIndexMap.contains(x) and
                opIndexMap[x] == opIndexMap[scOp]) {
              successorWriteOps.emplace_back(cast<vector::TransferWriteOp>(x));
            }
          }
          SmallVector<AffineExpr> exprs(loopType.getRank(), AffineExpr());
          bindSymbolsList<AffineExpr>(b.getContext(), exprs);
          SmallVector<Value> operands{inductionVars.begin(),
                                      inductionVars.end()};
          SmallVector<Value> smallRankShapeVars;

          auto getSmallRankShapeVars = [&](VectorType smallType) {
            size_t itrIdx = 0;
            SmallVector<bool> visitedAxis(rank, false);
            while ((int64_t)itrIdx < smallType.getRank()) {

              size_t endShape = getFirstTrueIndex(visitedAxis), dimSize = 1;
              if (endShape >= rank)
                llvm_unreachable("Invalid shape.");
              // skip non corresponding axis
              // e.g.: vector<32x16x1x32xbf16> -> vector<1x512x32xbf16>
              while (loopType.getShape()[endShape] >
                     smallType.getShape()[itrIdx]) {
                endShape++;
              }
              const size_t expandIdx = endShape;
              while (endShape < rank) {
                visitedAxis[endShape] = true;
                dimSize *= loopType.getShape()[endShape];
                if ((int64_t)dimSize == smallType.getShape()[itrIdx]) {
                  break;
                }
                endShape += 1;
              }
              const size_t expandSize = endShape - expandIdx + 1;
              AffineExpr calculateOffset;
              SmallVector<Value> offsetVars;

              for (size_t i = 0; i < expandSize; i++) {
                size_t startIdx = i + 1;
                size_t otherDimsSize = 1;
                while (startIdx < expandSize) {
                  otherDimsSize *= (loopType.getShape()[startIdx + expandIdx]);
                  startIdx++;
                }
                AffineExpr dimSize =
                    getAffineConstantExpr(otherDimsSize, b.getContext());
                if (i == 0) {
                  calculateOffset = exprs[i] * dimSize;
                } else {
                  calculateOffset = calculateOffset + exprs[i] * dimSize;
                }

                offsetVars.emplace_back(inductionVars[i + expandIdx]);
              }
              AffineMap map = AffineMap::get(0, expandSize, calculateOffset);

              Value offset =
                  b.createOrFold<affine::AffineApplyOp>(loc, map, offsetVars);
              smallRankShapeVars.emplace_back(offset);
              itrIdx++;
            }
          };

          if (loopType == sourceType) {
            getSmallRankShapeVars(destType);
          } else {
            getSmallRankShapeVars(sourceType);
          }

          auto padValue = b.create<arith::ConstantOp>(
              loc, b.getZeroAttr(loopType.getElementType()));

          SmallVector<bool> inBoundsVal(1, true);

          auto transferReadOp = b.create<vector::TransferReadOp>(
              loc,
              /*vectorType=*/kernelType,
              /*source=*/readSourceOp->getOperands()[0],
              /*indices=*/loopType == sourceType ? inductionVars
                                                 : smallRankShapeVars,
              /*padding=*/padValue,
              /*inBounds=*/inBoundsVal);

          SmallVector<Value> writeVars =
              loopType == sourceType ? smallRankShapeVars : inductionVars;
          SmallVector<Value> writeResults;
          for (auto successorWriteOp : successorWriteOps) {
            rectifyWriteOperationIndice(&successorWriteOp, writeVars);
            auto writeOp = b.create<vector::TransferWriteOp>(
                loc, transferReadOp->getResults()[0], loopState[0], writeVars,
                inBoundsVal);
            writeResults.emplace_back(writeOp->getResults()[0]);
          }
          maybeYieldValue(b, loc, writeResults);
        } else {
          // outter loop
          auto nxtFor = generateShapeCastReadWriteLoop(
              b, grpIdx, forDimIdx + 1, steps, loc, inductionVars, loopState);
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
}

void ForLoopGenerator::rectifyWriteOperationIndice(
    vector::TransferWriteOp *originalWriteOp,
    SmallVectorImpl<Value> &writeVars) {
  VectorType sucessWriteVectorType = originalWriteOp->getVectorType();
  ShapedType successWriteTensorType =
      cast<ShapedType>(originalWriteOp->getResultTypes()[0]);
  size_t inMutableIdx =
      successWriteTensorType.getRank() - sucessWriteVectorType.getRank();
  Operation::operand_range writeIndices = originalWriteOp->getIndices();

  for (size_t i = 0; i < inMutableIdx; i++)
    writeVars[i] = writeIndices[i];
}

void ForLoopGenerator::rectifyReadOperationIndice(
    vector::TransferReadOp *originalReadOp, VectorType loopType,
    ArrayRef<Value> inductionVars, SmallVectorImpl<Value> &readVars) {
  ShapedType readTensorType =
      cast<ShapedType>(originalReadOp->getSource().getType());
  // currently only broadcast (fuse as transfer_read) will move into more inner
  // loop
  if (readTensorType.getRank() - 1 >=
      (int64_t)getVectorBasedFusion().getOpAnchorPos()[*originalReadOp])
    return;

  int64_t itrIdx = loopType.getRank() - 1;
  int64_t readIdx = readTensorType.getRank() - 1;
  while (itrIdx >= 0 and readIdx >= 0) {
    if (readTensorType.getShape()[readIdx] == loopType.getShape()[itrIdx]) {
      readVars[readIdx] = inductionVars[itrIdx];
      readIdx--;
    }
    itrIdx--;
  }
}

/// generate transpose for loop
scf::ForOp LoopGeneratorImpl::generateShapeCastForLoop(const size_t grpIdx) {

  ShapeCastCanonicalizer &scCanonicalizer =
      getShapeCastCanonicalizers()[grpIdx];
  vector::ShapeCastOp &scOp = scCanonicalizer.getCandidateOps()[0];

  VectorType sourceType = scOp.getSourceVectorType();
  VectorType destType = scOp.getResultVectorType();
  DenseMap<Operation *, size_t> &opIndexMap =
      getVectorBasedFusion().getOpGroupIndexMap();

  OpBuilder b(scOp);
  SmallVector<Value> iterArgs;
  SmallVector<vector::TransferWriteOp> successorWriteOps;
  for (Operation *x : scOp->getUsers())
    if (isa<vector::TransferWriteOp>(x) and opIndexMap.contains(x) and
        opIndexMap[x] == opIndexMap[scOp])
      successorWriteOps.emplace_back(cast<vector::TransferWriteOp>(x));

  for (auto successorWriteOp : successorWriteOps)
    iterArgs.emplace_back(successorWriteOp->getOperands()[1]);

  SmallVector<Value> inductionVars;
  IRRewriter rewriter(scOp);
  const size_t groupStep = getVectorBasedFusion().getGroupMaxSteps()[grpIdx];

  bool isSourceMultiple =
      sourceType.getShape()[sourceType.getRank() - 1] % groupStep == 0;
  bool isDestMultiple =
      destType.getShape()[destType.getRank() - 1] % groupStep == 0;

  scf::ForOp forOp;
  bool canVectorizedLoadStore = isDestMultiple and isSourceMultiple and
                                scCanonicalizer.isReadWriteOnLastDim();
  if (canVectorizedLoadStore) {
    forOp = generateShapeCastReadWriteLoop(
        b, grpIdx, 0, groupStep, scOp.getLoc(), inductionVars, iterArgs);
  } else {
    // scalar data movement
    forOp = generateShapeCastReadWriteLoop(b, grpIdx, 0, 1, scOp.getLoc(),
                                           inductionVars, iterArgs);
  }
  for (auto [idx, successorWriteOp] : enumerate(successorWriteOps))
    rewriter.replaceOp(successorWriteOp, forOp->getResults()[idx]);

  rewriter.eraseOp(scOp);
  clearCurrentOperationGroup(grpIdx);
  return forOp;
}

/// mark which operation need to set correct for loop var idx
/// due to sometimes we need to chage for loop order like reduce operation.
void ForLoopGenerator::getCurrentGroupIndiceLoopMap(
    DenseMap<Operation *, DenseMap<size_t, size_t>> &indiceLoopMap,
    const size_t groupId, Operation *op,
    const DenseMap<size_t, size_t> &setIdxMap) {
  if (setIdxMap.empty()) {
    DenseMap<size_t, size_t> forIdxMap;
    VectorType groupVector =
        getVectorBasedFusion().getGroupBiggestRankVectorType()[groupId];
    for (size_t i = 0; (int64_t)i < groupVector.getRank(); i++) {
      forIdxMap[i] = i;
    }
    indiceLoopMap[op] = forIdxMap;
    return;
  }
  indiceLoopMap[op] = setIdxMap;
}

void ForLoopGenerator::clearCurrentOperationGroup(size_t grpIdx) {
  std::queue<Operation *>().swap(getVectorBasedFusion().getOpGroups()[grpIdx]);
};

scf::ForOp LoopGeneratorImpl::generateTransposeForLoop(const size_t grpIdx) {

  // transpose rank must bigger than 2
  TransposeCanonicalizer &tpCanonicalizer =
      getTransposeCanonicalizers()[grpIdx];
  vector::TransposeOp &tpOp = tpCanonicalizer.getCandidateOps()[0];
  IRRewriter rewriter(tpOp);

  VectorType vtType = tpOp.getResultVectorType();
  size_t rank = vtType.getRank();
  if (rank < 2) {
    llvm::llvm_unreachable_internal(
        "Wrong transpose operation appear. It's rank must bigger than 2.");
    return nullptr;
  }

  // permutation contains last dim can use optimizing algorithm
  ArrayRef<int64_t> permutation = tpOp.getPermutation();
  DenseSet<int64_t> permuteSet(permutation.begin(), permutation.end());
  bool isTwoDTranspose = tpCanonicalizer.isTwoDTranspose();

  Operation *successorWriteOp =
      getVectorBasedFusion()
          .getNextTargetOperationInCurrentGroup<vector::TransferWriteOp>(
              tpOp, grpIdx);

  DenseMap<Value, int> operandIdxMap;
  DenseMap<Value, Value> originalOperandMap, operandOriginalMap, resultIdxMap,
      forResultOrignalResultMap;
  SmallVector<Value> iterArgs;
  GenerateLoopHelper loopHelper(grpIdx, 0);
  prepareForLoopArgs(grpIdx, loopHelper);

  OpBuilder b(tpOp);
  int tpStep = TransposeCanonicalizer::TRANSPOSE_KERNEL::KERNEL_16X16;
  // only contains last dim can use fast transpose algorithm
  if ((tpCanonicalizer.getFirstTpIdx() == (rank - 1) or
       tpCanonicalizer.getSecondTpIdx() == (rank - 1)) and
      isTwoDTranspose) {
    scf::ForOp forOp = generateTransposeForLoopWithLastDim(
        b, tpStep, tpOp.getLoc(), successorWriteOp, loopHelper);

    rewriter.replaceOp(successorWriteOp, forOp);
    // clear current group operation
    clearCurrentOperationGroup(grpIdx);
    return forOp;
  }
  DenseMap<size_t, size_t> tpAxisMap;
  size_t itrIdx = 0;
  while (itrIdx < rank) {
    tpAxisMap[itrIdx] = permutation[itrIdx];
    itrIdx++;
  }
  // scalar data movement
  scf::ForOp forOp = generateTransposeScalarDataMovement(b, tpOp.getLoc(),
                                                         tpAxisMap, loopHelper);

  rewriter.replaceOp(successorWriteOp, forOp);
  clearCurrentOperationGroup(grpIdx);
  return forOp;
}

template <class T>
SmallVector<T, 4> &SpecialOperationCanonicalizer<T>::getCandidateOps() {
  return candidateRdOps;
};

void MultiReductionCanonicalizer::initReductionAxis() {
  auto reductionAxisRange = getCandidateOps()[0].getReductionDims();
  reductionAxis.assign(reductionAxisRange.begin(), reductionAxisRange.end());
  llvm::sort(reductionAxis);
}

void MultiReductionCanonicalizer::initParallelAxis() {
  llvm::SmallDenseSet<int64_t, 4> reductionAxisSet(reductionAxis.begin(),
                                                   reductionAxis.end());
  for (int64_t i = 0; i < typeRank; ++i)
    if (!reductionAxisSet.contains(i))
      parallelAxis.push_back(i);

  llvm::sort(parallelAxis);
}

int64_t MultiReductionCanonicalizer::getTypeRank() {
  auto srcRank = sourceType.getRank();
  typeRank = srcRank;
  return srcRank;
}

void MultiReductionCanonicalizer::getReductionAxisAndParallelAxis() {
  initReductionAxis();
  initParallelAxis();
}

bool MultiReductionCanonicalizer::hasLastDimReduction() {
  llvm::SmallDenseSet<int64_t, 4> reductionAxisSet(reductionAxis.begin(),
                                                   reductionAxis.end());
  bool res = false;
  if (reductionAxisSet.contains(typeRank - 1))
    res = true;

  haslastDimReduction = res;
  return res;
}

void MultiReductionCanonicalizer::prepareSpecialOperationInfo() {
  if (getCandidateOps().empty())
    return;

  sourceType = getCandidateOps()[0].getSourceVectorType();
  accType = cast<VectorType>(getCandidateOps()[0].getAcc().getType());
  getTypeRank();
  getReductionAxisAndParallelAxis();
  hasLastDimReduction();

  // whether all the reduction axis is 1
  for (auto axis : reductionAxis) {
    if (sourceType.getShape()[axis] != 1) {
      isEmptyReduction = false;
      break;
    }
  }
};

bool TransposeCanonicalizer::isTransposeOnAllOneDim() {
  vector::TransposeOp tpOp = getCandidateOps()[0];
  ArrayRef<int64_t> permutation = tpOp.getPermutation();
  VectorType tpVectorType = tpOp.getResultVectorType();
  int64_t itrIdx = 0;
  while (itrIdx < tpVectorType.getRank()) {
    if (itrIdx == permutation[itrIdx]) {
      itrIdx++;
      continue;
    }
    if (tpVectorType.getShape()[itrIdx] != 1)
      return false;

    itrIdx++;
  }
  return true;
}

bool TransposeCanonicalizer::isTwoDTranspose() {
  ArrayRef<int64_t> permutation = getCandidateOps()[0].getPermutation();

  size_t rank = permutation.size();
  int diffCount = 0;
  // get the first transpose axis
  size_t itrIdx = 0;
  while (itrIdx < rank) {
    if ((int64_t)itrIdx != permutation[itrIdx])
      diffCount += 1;

    itrIdx += 1;
  }

  itrIdx = 0;
  while (itrIdx < rank) {
    if (permutation[itrIdx] != (int64_t)itrIdx) {
      firstTpIdx = itrIdx;
      break;
    }
    itrIdx++;
  }

  itrIdx = 0;
  // get the second transpose axis
  while (itrIdx < rank) {
    if (permutation[itrIdx] == (int64_t)firstTpIdx) {
      secondTpIdx = itrIdx;
      break;
    }
    itrIdx++;
  }

  const int tpStep = TRANSPOSE_KERNEL::KERNEL_16X16;
  VectorType vtType = getCandidateOps()[0].getResultVectorType();
  // currently we only support shape that is an integer multiple of tpStep
  if (vtType.getShape()[getFirstTpIdx()] % tpStep != 0 or
      vtType.getShape()[getSecondTpIdx()] % tpStep != 0)
    return false;

  return diffCount == 2;
}

bool TransposeCanonicalizer::transposeOnLastDim() {
  ArrayRef<int64_t> permutation = getCandidateOps()[0].getPermutation();
  size_t rank = permutation.size();
  if (permutation[rank - 1] != (int64_t)rank - 1)
    return false;

  VectorType vtType = getCandidateOps()[0].getResultVectorType();
  return vtType.getShape()[rank - 1] % getVectorStep() == 0;
}

bool ShapeCastCanonicalizer::isReadWriteOnLastDim() {
  vector::ShapeCastOp &shapeCastOp = getCandidateOps()[0];
  VectorType sourceType = shapeCastOp.getSourceVectorType();
  VectorType destType = shapeCastOp.getResultVectorType();
  VectorType smallRankType =
      sourceType.getRank() > destType.getRank() ? destType : sourceType;
  VectorType largeRankType =
      sourceType.getRank() < destType.getRank() ? destType : sourceType;
  SmallVector<bool> visitedAxis(largeRankType.getRank(), false);
  // Map the index of the larger rank shape to the index of the smaller rank
  // shape.
  DenseMap<size_t, SmallVector<size_t>> shapeIdxMap;
  for (size_t i = 0; (int64_t)i < smallRankType.getRank(); i++)
    shapeIdxMap[i] = SmallVector<size_t>();

  int64_t itrIdx = 0;
  while (itrIdx < smallRankType.getRank()) {
    int64_t endShape = getFirstTrueIndex(visitedAxis), dimSize = 1;
    if (endShape >= largeRankType.getRank() or endShape < 0)
      llvm_unreachable("Invalid endShape.");

    // skip non corresponding axis
    // e.g.: vector<32x16x1x32xbf16> -> vector<1x512x32xbf16>
    while (largeRankType.getShape()[endShape] >
           smallRankType.getShape()[itrIdx])
      endShape++;

    while (endShape < largeRankType.getRank()) {
      visitedAxis[endShape] = true;
      shapeIdxMap[itrIdx].emplace_back(endShape);
      dimSize *= largeRankType.getShape()[endShape];
      if ((int64_t)dimSize == smallRankType.getShape()[itrIdx])
        break;

      endShape++;
    }
    itrIdx++;
  }
  // check if the last dim is read write
  SmallVector<size_t> lastDims = shapeIdxMap[smallRankType.getRank() - 1];
  DenseSet<size_t> set(lastDims.begin(), lastDims.end());
  return set.contains(largeRankType.getRank() - 1);
}

template <class T>
void addDummyInit(SmallVector<T, 8> &canonicalizer, size_t steps = 1) {
  canonicalizer.emplace_back(T({}, steps));
};

void LoopGeneratorImpl::clearSpecialOperationCanonicalizers() {
  getMultiRdCanonicalizers().clear();
  getBroadcastCanonicalizers().clear();
  getTransposeCanonicalizers().clear();
  getShapeCastCanonicalizers().clear();
}

void LoopGeneratorImpl::dummyInitSpecialOperation(size_t steps) {
  addDummyInit<MultiReductionCanonicalizer>(getMultiRdCanonicalizers(), steps);
  addDummyInit<BroadcastCanonicalizer>(getBroadcastCanonicalizers(), steps);
  addDummyInit<TransposeCanonicalizer>(getTransposeCanonicalizers(), steps);
  addDummyInit<ShapeCastCanonicalizer>(getShapeCastCanonicalizers(), steps);
}

void LoopGeneratorImpl::initSpeicalOperationCanonicalizers() {
  clearSpecialOperationCanonicalizers();
  SmallVector<std::queue<Operation *>, 8> &opGroups =
      getVectorBasedFusion().getOpGroups();
  for (auto [idx, grp] : llvm::enumerate(opGroups)) {
    dummyInitSpecialOperation(getVectorBasedFusion().getGroupMaxSteps()[idx]);
    if (grp.empty())
      continue;

    std::queue<Operation *> tempQ(grp);
    while (!tempQ.empty()) {
      auto op = tempQ.front();
      tempQ.pop();
      TypeSwitch<Operation *>(op)
          .Case<vector::MultiDimReductionOp>([&](vector::MultiDimReductionOp
                                                     multiReductionOp) {
            getMultiRdCanonicalizers().back().getCandidateOps().emplace_back(
                cast<vector::MultiDimReductionOp>(op));
            getMultiRdCanonicalizers().back().prepareSpecialOperationInfo();
          })
          .Case<vector::TransposeOp>([&](vector::TransposeOp tpOp) {
            getTransposeCanonicalizers().back().getCandidateOps().emplace_back(
                cast<vector::TransposeOp>(op));
          })
          .Case<vector::ShapeCastOp>([&](vector::ShapeCastOp spOp) {
            getShapeCastCanonicalizers().back().getCandidateOps().emplace_back(
                cast<vector::ShapeCastOp>(op));
          })
          .Default([&](Operation *op) {});
    }
  }
}

template <class T, class U>
void LoopGeneratorImpl::processSpecialOperation(
    T &canonicalizers, const std::function<void(const size_t)> &generateFunc) {
  for (auto [groupId, canonicalizer] : llvm::enumerate(canonicalizers)) {
    SmallVector<U, 4> &ops = canonicalizer.getCandidateOps();
    if (!ops.empty())
      // generate MultiReduction for loops
      generateFunc(groupId);
  }
}

void LoopGeneratorImpl::canonicalizeSpecialOperation() {

  initSpeicalOperationCanonicalizers();
  // traverse all groups
  llvm::SmallVector<MultiReductionCanonicalizer, 8> &multiRdCanonicalizers =
      getMultiRdCanonicalizers();
  processSpecialOperation<SmallVector<MultiReductionCanonicalizer, 8>,
                          vector::MultiDimReductionOp>(
      multiRdCanonicalizers, [this](const size_t grpIdx) {
        (void)generateMultiReductionForLoop(grpIdx);
      });
  // generate loop for transpose operation
  SmallVector<TransposeCanonicalizer, 8> &transposeCanonicalizers =
      getTransposeCanonicalizers();
  processSpecialOperation<SmallVector<TransposeCanonicalizer, 8>,
                          vector::TransposeOp>(
      transposeCanonicalizers,
      [this](const size_t grpIdx) { (void)generateTransposeForLoop(grpIdx); });
  // generate loop for shapecast opearation
  SmallVector<ShapeCastCanonicalizer, 8> &shapeCastCanonicalizers =
      getShapeCastCanonicalizers();
  processSpecialOperation<SmallVector<ShapeCastCanonicalizer, 8>,
                          vector::ShapeCastOp>(
      shapeCastCanonicalizers,
      [this](const size_t grpIdx) { (void)generateShapeCastForLoop(grpIdx); });
}

void VectorOperationCanonicalizer::run() {
  auto &fusionStrategy = fusion.getGroupOperationFusion();
  if (kind == CanonicalizerKind::GroupOperations) {
    fusion.run();
    // 1. Analysis the operation's operands and results
    // We need to analyze which operation's result is needed by other
    // operations, and we need to pass these results correctly. Mapping the
    // operation result value with the forloop yeild result value. We can
    // replace the operation operand as: map(operand, forloop yield result) ->
    // operand = loop yield result We put all the operation result into this
    // map.

    // 1.a. Find results which should be generated by current group for
    // using as operands to other operations?

    // Traverse all operations. If the operand of operations in other groups
    // or outside the group is the result of the operation in current group,
    // then the current operation needs to generate a result. We use `setvector`
    // to save the results that need to be generated by the current group.

    //  1.b. What operands are needed to find in the current group, and where
    //  can they be obtained ?

    //  Thanks to 1.a, we get the result generated by the operations of
    //  each group, and this result will use `scf.yield` to generate a
    //  new result. Since the scope of the parent block of mlir is covered
    //  the current operation, the current operation does not need to pass
    //  these `for loop result` to the `iterArgs` of the required `for loop`.
    //  It only needs to replace the operand of the current operation with the
    //  corresponding `for loop yield result`.

    // However, for some operations that are not DPS, we need to canonicalize
    // them. Canonicalization means that the operand of this operation is a
    // vector but we can't get this vector due to it locates in another block
    // which has a different scope. Therefore, it is necessary to write the
    // vector results into a temporary tensor to save it. Then the vector
    // needs to be read from the tensor before the current operation operate
    // on it. Therefore,  `empty tensor`, `transfer_write` and `transfer_read`
    // need to be inserted at target place.
    if (enableDebugPrinter) {
      printGroupOps(fusion.getGroupOperationFusion().getOpGroups());
      LDBG("___________ before analysis ________________");
    }
    fusion.canonicalizeEachOperationGroup();
    if (enableDebugPrinter) {
      LDBG("___________ after analysis ________________");
      printGroupOps(fusion.getGroupOperationFusion().getOpGroups());
    }

    loopGenerator.setVectorBaseFusion(fusion.getGroupOperationFusion());
    // Speical Operation Canonicalization
    loopGenerator.canonicalizeSpecialOperation();

    // 2.Generate vectorized IR for each operation group
    for (size_t idx = 0; idx < fusionStrategy.getOpGroups().size(); ++idx)
      loopGenerator.generateGroupOpVectorizedIR(idx);

    // 3. Some IR cleanup work
    DominanceInfo domInfo;
    eliminateCommonSubExpressions(
        rewriter, domInfo, loopGenerator.getVectorBasedFusion().getFunction());
  } else {
    // TODO: need to add directly canonicalize operations logic
    // generateGroupOpVectorizedIR(idx, grp, fusionStrategy.opGroupIndexMap);
  }
}

///
void ForLoopGenerator::setOperationCorrectOperand(
    Operation *op, const DenseMap<Operation *, AffineMap> &opPermuationMap,
    GenerateLoopHelper &loopHelperParam) {
  for (auto [idx, opd] : llvm::enumerate(op->getOperands())) {
    if (not loopHelperParam.originalOperandLoopArgsMap.contains(opd))
      continue;

    Value loopArg = loopHelperParam.originalOperandLoopArgsMap[opd];
    if (not loopHelperParam.currentLoopStateIdxMap.contains(loopArg))
      continue;

    op->setOperand(
        idx,
        loopHelperParam
            .loopIterArgs[loopHelperParam.currentLoopStateIdxMap.at(loopArg)]);
  }
  int offset = isa<vector::TransferWriteOp>(op) ? 2 : 1;
  if (dyn_cast<vector::TransferWriteOp>(op) ||
      dyn_cast<vector::TransferReadOp>(op)) {
    if (not opPermuationMap.contains(op))
      llvm_unreachable("Map must contains operation.");

    auto permutationMap = opPermuationMap.at(op);

    auto dimExpr = permutationMap.getResults();
    for (auto [idx, x] : llvm::enumerate(dimExpr)) {

      if (not isa<AffineDimExpr, AffineConstantExpr>(x))
        llvm::llvm_unreachable_internal(
            "Permuatation map must contains dim expr.");

      int64_t dim = 0;
      if (auto d = dyn_cast<AffineDimExpr>(x)) {
        dim = d.getPosition();
      } else if (auto d = dyn_cast<AffineConstantExpr>(x)) {
        dim = d.getValue();
      }

      ShapedType tensorType =
          cast<ShapedType>(op->getOperandTypes()[offset - 1]);
      int64_t varIdx = dim;
      if (tensorType.getRank() >
          (int64_t)loopHelperParam.inductionVars.size()) {
        int64_t tensorOffset =
            tensorType.getRank() - loopHelperParam.inductionVars.size();
        if (dim < tensorOffset)
          continue;

        varIdx = dim - tensorOffset;
      }
      if (loopHelperParam.indiceLoopMap.contains(op))
        op->setOperand(
            dim + offset,
            loopHelperParam
                .inductionVars[loopHelperParam.indiceLoopMap[op][varIdx]]);
      else
        op->setOperand(dim + offset, loopHelperParam.inductionVars[varIdx]);
    }
    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      size_t grpIdx = getVectorBasedFusion().getOpGroupIndexMap()[op];
      VectorType loopType =
          getVectorBasedFusion().getGroupBiggestRankVectorType()[grpIdx];
      SmallVector<Value> readIndices(readOp.getIndices().begin(),
                                     readOp.getIndices().end());
      rectifyReadOperationIndice(&readOp, loopType,
                                 loopHelperParam.inductionVars, readIndices);
      readOp.getIndicesMutable().assign(readIndices);
    }
  }
}

scf::ForOp ForLoopGenerator::constructNestedForOp(
    const size_t groupIdx, OpBuilder &b, const Location &loc,
    ArrayRef<int64_t> dims, GenerateLoopHelper &loopHelper) {
  const int loop_step = getVectorBasedFusion().getGroupMaxSteps()[groupIdx];
  // loop initialization variable
  auto zero = makeIndexArithConstantOp(b, loc, 0);
  auto forSteps = makeIndexArithConstantOp(
      b, loc, loopHelper.anchorIdx == dims.size() - 1 ? loop_step : 1);
  auto numIter = makeIndexArithConstantOp(b, loc, dims[loopHelper.anchorIdx]);

  // Create a loop and move vectorized operation into loops.
  auto forOp = b.create<scf::ForOp>(
      loc, zero, numIter, forSteps, loopHelper.loopIterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        loopHelper.inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (loopHelper.anchorIdx == dims.size() - 1) {
          std::queue<Operation *> &opQueue =
              getVectorBasedFusion().getOpGroups()[groupIdx];
          loopHelper.loopIterArgs = loopState;
          // 1. get operations in current anchor position
          std::queue<Operation *> movingOperation;
          getOperationInCurrentAnchor(loopHelper.anchorIdx, opQueue,
                                      movingOperation);

          // 2. rewrite operation as vectorize IR
          rewriteOperationAsVectorize(b, groupIdx, &movingOperation);

          // 3. move opeartions to current for block
          moveOperationsToCurrentForBody(b, movingOperation, loopHelper);

          getResultInCurrentOps(loopHelper.anchorIdx, groupIdx, movingOperation,
                                loopHelper.nextAnchorResults,
                                loopHelper.nextAnchorResultsIdxMap,
                                loopHelper.nextAnchorResultOrignalResultMap);
          maybeYieldValue(b, loc, loopHelper.nextAnchorResults);
        } else {
          // outter loop

          // 1. move pre-Op to current body
          DenseMap<Value, int> nextAnchorArgsIdxMap;
          SmallVector<Value, 4> nextAnchorArgs;
          DenseMap<Value, Value> currentOriginalOperandMap =
              loopHelper.originalOperandLoopArgsMap;
          DenseMap<Value, Value> currentOperandOriginalMap =
              loopHelper.loopArgsOriginalOperandMap;
          DenseMap<Value, int> currentArgsIdxMap =
              loopHelper.currentLoopStateIdxMap;

          std::queue<Operation *> movedQueue;
          std::queue<Operation *> &opQueue =
              getVectorBasedFusion().getOpGroups()[groupIdx];
          SmallVector<Value> tmpArgs(loopState);
          loopHelper.updateDataBeforePreOpMove(tmpArgs, opQueue, movedQueue);
          movePreOpToCurrentAnchor(b, nextAnchorArgsIdxMap, nextAnchorArgs,
                                   loopHelper);
          loopHelper.updateDataAfterPreOpMove(nextAnchorArgsIdxMap,
                                              nextAnchorArgs);
          loopHelper.anchorIdx += 1;
          auto nxtFor =
              constructNestedForOp(groupIdx, b, loc, dims, loopHelper);
          loopHelper.anchorIdx -= 1;
          SmallVector<Value, 4> currentArgs(loopState);

          loopHelper.updateCurrentArgsStatus(currentArgsIdxMap, currentArgs,
                                             currentOriginalOperandMap,
                                             currentOperandOriginalMap);

          loopHelper.updateDataBeforePostOpMove(
              tmpArgs, currentArgsIdxMap, currentOriginalOperandMap,
              currentOperandOriginalMap, nxtFor->getResults(), b.getBlock(),
              movedQueue, loopHelper.anchorIdx);
          movePostOpToCurrentAnchor(b, loopHelper);

          generateLoopResults(b, loc, loopHelper, nextAnchorArgsIdxMap);

          maybeYieldValue(b, loc, loopHelper.nextAnchorResults);
        }
      });
  return forOp;
}

Value setOutGroupOperationOperandResult(Operation *op,
                                        const VectorType &newOperandType) {
  auto ret =
      TypeSwitch<Operation *, Value>(op)
          .Case<arith::ConstantOp>([&](arith::ConstantOp constantOp) {
            IRRewriter rewriter(op);
            rewriter.setInsertionPointAfter(op);
            Type resultElementType = newOperandType.getElementType();
            auto value = constantOp.getValue();
            Attribute initValueAttr;

            if (isa<ElementsAttr>(value)) {
              auto valueType = mlir::dyn_cast<ElementsAttr>(value);
              if (valueType.isSplat()) {
                if (isa<FloatType>(valueType.getElementType()))
                  initValueAttr = FloatAttr::get(
                      resultElementType,
                      valueType.getSplatValue<APFloat>().convertToDouble());
                else
                  initValueAttr = IntegerAttr::get(
                      resultElementType,
                      valueType.getSplatValue<APInt>().getSExtValue());
              } else {
                // write original vector into tensor
                // then we transfer_read from the tensor
                llvm_unreachable("Not support non-splat constant value.");
              }
            } else if (isa<FloatType>(resultElementType)) {
              initValueAttr = FloatAttr::get(
                  resultElementType, cast<FloatAttr>(value).getValueAsDouble());
            } else {
              initValueAttr = IntegerAttr::get(
                  resultElementType, cast<IntegerAttr>(value).getInt());
            }

            auto cntOp = rewriter.create<arith::ConstantOp>(
                rewriter.getUnknownLoc(),
                DenseElementsAttr::get(newOperandType, {initValueAttr}));
            return cntOp->getResults()[0];
          })
          .Default([&](Operation *op) { return Value(); });
  return ret;
}

void setOperationOperandResult(Operation *op, const VectorType &newOperandType,
                               const DenseMap<Operation *, size_t> &opMap) {
  for (auto [idx, x] : llvm::enumerate(op->getOperands())) {
    if (dyn_cast<VectorType>(x.getType())) {
      if (!opMap.contains(x.getDefiningOp())) {
        auto result = setOutGroupOperationOperandResult(x.getDefiningOp(),
                                                        newOperandType);
        op->setOperand(idx, result);
      } else {
        x.setType(newOperandType);
      }
    }
  }
  for (auto x : op->getResults())
    if (dyn_cast<VectorType>(x.getType()))
      x.setType(newOperandType);
};

void ForLoopGenerator::createNewConstantOp(
    Operation *srcOp, vector::TransferWriteOp *transferWriteOp,
    size_t groupSteps) {
  DenseMap<Operation *, AffineMap> &opPermuationMap =
      getVectorBasedFusion().getOpPermuationMap();

  IRRewriter srcWriter(srcOp);
  VectorType newOperandType =
      getVectorBasedFusion().getTypeHelper().getVectorzedType(
          cast<Operation *>(srcOp), groupSteps);
  auto srcConstantOp = dyn_cast<arith::ConstantOp>(srcOp);
  Operation *newConstantOp;
  if (isa<DenseElementsAttr>(srcConstantOp.getValue())) {
    auto valueType = dyn_cast<DenseElementsAttr>(srcConstantOp.getValue());
    if (valueType.isSplat()) {
      FailureOr<Value> res = createArithSplatConstantOp(
          srcWriter, srcOp->getLoc(), valueType, newOperandType);
      if (failed(res)) {
        llvm::llvm_unreachable_internal("Wrong to create constant op.");
      }
      newConstantOp = res.value().getDefiningOp();
    } else {
      // TODO: need to test not splat value
      llvm::llvm_unreachable_internal(
          "Can't support not splat constant value.");
    }

    newConstantOp->getResult(0).setType(newOperandType);
    transferWriteOp->setOperand(0, newConstantOp->getResult(0));
    opPermuationMap.insert(
        {*transferWriteOp, transferWriteOp->getPermutationMap()});
    setOpVectorizationPermutationMap(
        *transferWriteOp, srcWriter,
        cast<ShapedType>(transferWriteOp->getResults()[0].getType()),
        transferWriteOp->getPermutationMap());
    return;
  }
  llvm::llvm_unreachable_internal(
      "Can't support not DenseElementsAttr constant.");
}

/// Rewrite the operations in the group to vectorized form.
void ForLoopGenerator::rewriteOperationAsVectorize(
    OpBuilder &rewriter, size_t groupId, const std::queue<Operation *> *queue) {
  const std::queue<Operation *> groupOps =
      !queue ? getVectorBasedFusion().getOpGroups()[groupId] : *queue;

  const DenseMap<Operation *, size_t> &opMap =
      getVectorBasedFusion().getOpGroupIndexMap();
  DenseMap<Operation *, AffineMap> &opPermuationMap =
      getVectorBasedFusion().getOpPermuationMap();
  std::queue<Operation *> transformQueue(groupOps);
  size_t groupSteps = getVectorBasedFusion().getGroupMaxSteps()[groupId];

  while (!transformQueue.empty()) {
    Operation *op = transformQueue.front();
    transformQueue.pop();
    VectorType newOperandType =
        getVectorBasedFusion().getTypeHelper().getVectorzedType(op, groupSteps);
    auto lowerResult =
        TypeSwitch<Operation *, LogicalResult>(op)
            .Case<vector::TransferWriteOp>(
                [&](vector::TransferWriteOp transferWriteOp) {
                  IRRewriter rewriter(transferWriteOp);

                  Operation *srcOp =
                      transferWriteOp->getOperand(0).getDefiningOp();
                  if (isa<arith::ConstantOp>(srcOp)) {
                    createNewConstantOp(srcOp, &transferWriteOp, groupSteps);
                  } else {
                    opPermuationMap.insert(
                        {transferWriteOp, transferWriteOp.getPermutationMap()});
                    transferWriteOp->getOperand(0).setType(newOperandType);

                    setOpVectorizationPermutationMap(
                        transferWriteOp, rewriter,
                        cast<ShapedType>(
                            transferWriteOp->getResult(0).getType()),
                        transferWriteOp.getPermutationMap());
                  }

                  return success();
                })
            .Case<vector::TransferReadOp>(
                [&](vector::TransferReadOp transferReadOp) {
                  opPermuationMap.insert(
                      {transferReadOp, transferReadOp.getPermutationMap()});
                  transferReadOp->getResult(0).setType(newOperandType);
                  setOpVectorizationPermutationMap(
                      transferReadOp, rewriter,
                      cast<ShapedType>(transferReadOp.getSource().getType()),
                      transferReadOp.getPermutationMap());

                  return success();
                })
            .Case<vector::MultiDimReductionOp>(
                [&](vector::MultiDimReductionOp multiReductionOp) {
                  multiReductionOp.dump();
                  llvm::llvm_unreachable_internal(
                      "It should not appear this operation.");
                  return failure();
                })
            .Case<ARITH_CAST_OPERATIONS>([&](Operation *extfOp) {
              extfOp->getResult(0).setType(newOperandType);
              return success();
            })
            .Default([&](Operation *op) {
              if (isSpecialOp(op)) {
                op->dump();
                llvm::llvm_unreachable_internal(
                    "It should not appear this operation.");
                return failure();
              }
              setOperationOperandResult(op, newOperandType, opMap);
              return success();
            });
    if (failed(lowerResult)) {
      LDBG("Failed to rewrite operation: " << *op << "\n");
      llvm_unreachable("Failed to rewrite operation");
    }
  }
}

void GroupOperationFusionImpl::removeOpInCurrentGroups(size_t grpIdx,
                                                       Operation *op,
                                                       Operation *replacedOp) {
  std::queue<Operation *> tmpOpQueue(
      getGroupOperationFusion().getOpGroups()[grpIdx]);
  std::queue<Operation *> newOpQueue;
  while (!tmpOpQueue.empty()) {
    auto curOp = tmpOpQueue.front();
    tmpOpQueue.pop();
    if (curOp != op) {
      newOpQueue.push(curOp);
      continue;
    }
    getGroupOperationFusion().getOpGroupIndexMap().erase(curOp);
    getGroupOperationFusion().getOpAnchorPos().erase(curOp);
  }
  getGroupOperationFusion().getOpGroups()[grpIdx] = newOpQueue;

  // erase and replace the operation
  SmallVector<Operation *> usesOp(op->getUsers().begin(), op->getUsers().end());
  IRRewriter rewriter(op);
  rewriter.replaceOp(op, replacedOp);
  // update removed operation related operation anchor position
  // getGroupOperationFusion().getOpAnchorPos()[replacedOp] =
  //     getOperationMaxVectorType(replacedOp)->getRank() - 1;
  // for (Operation *x : usesOp)
  //   getGroupOperationFusion().getOpAnchorPos()[x] =
  //       getOperationMaxVectorType(x)->getRank() - 1;

  // updateOpGroupInfo(grpIdx);
}

void GroupOperationFusionImpl::updateOpGroupInfo(size_t grpIdx) {
  std::queue<Operation *> tmpOpQueue(
      getGroupOperationFusion().getOpGroups()[grpIdx]);
  // dummy init
  VectorType currentMaxRankType =
      getOperationMaxVectorType(tmpOpQueue.front()).value();
  getGroupOperationFusion().getGroupBiggestRankVectorType()[grpIdx] =
      currentMaxRankType;

  while (!tmpOpQueue.empty()) {
    auto curOp = tmpOpQueue.front();
    tmpOpQueue.pop();
    VectorType type = getOperationMaxVectorType(curOp).value();
    if (type.getRank() > currentMaxRankType.getRank())
      getGroupOperationFusion().getGroupBiggestRankVectorType()[grpIdx] = type;
  }
}

void GroupOperationFusionImpl::updateOpOperandResultInGroups(
    size_t opGid, Operation *op, const Value &init, const Value &result) {
  std::queue<Operation *> tmpOpQueue(
      getGroupOperationFusion().getOpGroups()[opGid]);
  std::queue<Operation *> newOpQueue;
  while (!tmpOpQueue.empty()) {
    auto curOp = tmpOpQueue.front();
    tmpOpQueue.pop();

    if (curOp != op) {
      newOpQueue.push(curOp);
      continue;
    }

    if (!failed(getOperationVectorType(init.getDefiningOp()))) {
      newOpQueue.push(init.getDefiningOp());
      getGroupOperationFusion().getOpGroupIndexMap()[init.getDefiningOp()] =
          opGid;
      getGroupOperationFusion().getOpAnchorPos()[init.getDefiningOp()] =
          getGroupOperationFusion().getOpAnchorPos()[op];
    }
    newOpQueue.push(op);

    if (result && !failed(getOperationVectorType(result.getDefiningOp()))) {
      newOpQueue.push(result.getDefiningOp());
      getGroupOperationFusion().getOpGroupIndexMap()[result.getDefiningOp()] =
          opGid;
      getGroupOperationFusion().getOpAnchorPos()[result.getDefiningOp()] =
          getGroupOperationFusion().getOpGroupIndexMap()[op];
    }
  }
  getGroupOperationFusion().getOpGroups()[opGid] = newOpQueue;
}

void GroupOperationFusionImpl::generateEmptyTensorAndWrite(
    Operation *sourceOp,
    DenseMap<Operation *, std::pair<Value, Value>> &srcOpCanoniclizedMap,
    size_t anchorPos, ReturnTypeKind retKind,
    DenseMap<Operation *, size_t> &visitedOperation) {
  DenseMap<Operation *, size_t> &opGroupIndexMap =
      getGroupOperationFusion().getOpGroupIndexMap();
  SmallVector<SetVector<Value>, 8> &groupOpInitArgs =
      getGroupOperationFusion().getGroupOpInitArgs();
  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
      &groupOpResults = getGroupOperationFusion().getGroupOpResults();
  size_t sourceOpGid = opGroupIndexMap[sourceOp];

  auto [tsr, writeOpresult] =
      canonicalizeSourceOperation(sourceOp, visitedOperation);
  auto writeOp = writeOpresult.getDefiningOp<vector::TransferWriteOp>();
  srcOpCanoniclizedMap.insert({sourceOp, {tsr, writeOpresult}});
  updateOpOperandResultInGroups(sourceOpGid, sourceOp, tsr, writeOpresult);
  groupOpInitArgs[sourceOpGid].insert(tsr);
  groupOpResults[sourceOpGid].insert({writeOpresult, {retKind, anchorPos}});
  // write opeartion anchor pos is same with current operation
  getGroupOperationFusion().getOpAnchorPos()[writeOp] =
      writeOp.getVectorType().getRank() - 1;
  getGroupOperationFusion().getOpPermuationMap()[writeOp] =
      writeOp.getPermutationMap();
}

void GroupOperationFusionImpl::specialOperationRectify(
    DenseMap<Operation *, size_t> &visitedOperation) {
  auto &opGroups = getGroupOperationFusion().getOpGroups();
  IRRewriter rewriter(getGroupOperationFusion().getFunction());

  for (auto [idx, grp] : llvm::enumerate(opGroups)) {
    std::queue<Operation *> tmpQueue(grp);
    std::queue<Operation *> newQueue;
    while (!tmpQueue.empty()) {
      auto op = tmpQueue.front();
      tmpQueue.pop();
      //  remain transfer read operation to do the broadcast fusion
      if (isa<vector::BroadcastOp>(op)) {
        auto srcOp = op->getOperand(0).getDefiningOp();
        if (not isa<vector::TransferReadOp>(srcOp))
          llvm_unreachable("Must be read operation.");

        // only have write operation, otherwise the group size will bigger
        // than 1. Because the last operation is always a write operation in
        // each group
        getGroupOperationFusion().getOpAnchorPos()[srcOp] =
            getGroupOperationFusion().getOpAnchorPos()[op];

        rewriter.replaceOp(op, srcOp);
        continue;
      }
      // anchor of multidim reduction rectify
      if (isa<vector::MultiDimReductionOp>(op)) {
        auto accSourceOp = op->getOperand(1).getDefiningOp();
        getGroupOperationFusion().getOpAnchorPos()[accSourceOp] =
            getOperationVectorType(accSourceOp)->getRank() - 1;
      }
      newQueue.push(op);
    }
    getGroupOperationFusion().getOpGroups()[idx] = newQueue;
  }
}

void GroupOperationFusionImpl::updateReturnResultKind(Operation *sourceOp,
                                                      size_t sourceOpGid,
                                                      ReturnTypeKind rtKind) {
  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
      &groupOpResults = getGroupOperationFusion().getGroupOpResults();
  DenseMap<Operation *, size_t> &OpAnchorPos =
      getGroupOperationFusion().getOpAnchorPos();

  Value sourceResult = sourceOp->getResults()[0];
  if (srcOpCanoniclizedMap.contains(sourceOp))
    sourceResult = srcOpCanoniclizedMap[sourceOp].second;

  size_t srcOpAnchor = groupOpResults[sourceOpGid][sourceResult].second;
  ReturnTypeKind prevRtKind = groupOpResults[sourceOpGid][sourceResult].first;
  srcOpAnchor = std::min(srcOpAnchor, OpAnchorPos[sourceOp]);
  if (prevRtKind != rtKind) {
    groupOpResults[sourceOpGid][sourceResult] =
        std::make_pair(ReturnTypeKind::RT_Both, srcOpAnchor);
    return;
  }
  if (rtKind == ReturnTypeKind::RT_InGroup)
    groupOpResults[sourceOpGid][sourceResult] =
        std::make_pair(rtKind, srcOpAnchor);
}

void GroupOperationFusionImpl::replaceConstantOpAsNewOp(Operation *op,
                                                        Operation *sourceOp,
                                                        size_t operandIdx) {
  DenseMap<Operation *, size_t> &opGroupIndexMap =
      getGroupOperationFusion().getOpGroupIndexMap();
  if (!opGroupIndexMap.contains(op)) {
    return;
  }
  // TODO: add more operation to this case, write a constant value need
  // to do this
  if (isa<vector::TransferWriteOp>(op) and operandIdx == 0)
    return;

  if (isa<vector::MultiDimReductionOp>(op)) {
    if (operandIdx == 1) {
      // accumulate value, just empty tensor is okay
      auto resultTensor = getOperationResultTensor(sourceOp, visitedOperation);
      auto opInit = canonicalizeCurrentOperation(op, resultTensor, operandIdx);
      updateOpOperandResultInGroups(opGroupIndexMap[op], op, opInit);
      return;
    }
    // source operation is the value
    llvm::llvm_unreachable_internal(
        "Need to add reduce constant operation optimization.");
  }

  auto constantOp = cast<arith::ConstantOp>(sourceOp);
  IRRewriter rewriter(constantOp);
  size_t groupSteps =
      getGroupOperationFusion().getGroupMaxSteps()[opGroupIndexMap[op]];

  if (isa<DenseElementsAttr>(constantOp.getValue())) {
    VectorType newOperandType =
        getGroupOperationFusion().getTypeHelper().getVectorzedType(op,
                                                                   groupSteps);
    auto valueType = cast<DenseElementsAttr>(constantOp.getValue());
    if (valueType.isSplat()) {
      FailureOr<Value> res = createArithSplatConstantOp(
          rewriter, constantOp->getLoc(), valueType, newOperandType);
      if (failed(res))
        llvm::llvm_unreachable_internal("Wrong to create constant op.");

      op->setOperand(operandIdx, res.value());
      // transfer read operation just use the constant value to do
      // calculation, don't need to read.
      if (isa<vector::TransferReadOp>(op) and operandIdx == 0)
        removeOpInCurrentGroups(opGroupIndexMap[op], op,
                                op->getOperand(0).getDefiningOp());
      return;
    }
    llvm::llvm_unreachable_internal("Can't support not splat constant value.");
  }
}

void GroupOperationFusionImpl::makeSourceOpWriteResultToTensor(
    Operation *sourceOp, size_t sourceOpGid, ReturnTypeKind rtKind) {
  DenseMap<Operation *, size_t> &OpAnchorPos =
      getGroupOperationFusion().getOpAnchorPos();
  SmallVector<SetVector<Value>, 8> &groupOpInitArgs =
      getGroupOperationFusion().getGroupOpInitArgs();
  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
      &groupOpResults = getGroupOperationFusion().getGroupOpResults();

  if (!srcOpCanoniclizedMap.contains(sourceOp)) {
    // get write operation
    if (Operation *writeOp =
            getGroupOperationFusion()
                .getNextTargetOperationInCurrentGroup<vector::TransferWriteOp>(
                    sourceOp, sourceOpGid)) {
      auto writeOpresult = writeOp->getResults()[0];
      auto originalWriteTensor = writeOp->getOperands()[1];
      // find original tensor.empty operation
      Value writeTensor =
          findOriginalTensor(originalWriteTensor, sourceOp->getBlock());
      if (writeTensor != originalWriteTensor)
        getGroupOperationFusion()
            .getOperandOriginalValue()[originalWriteTensor] = writeTensor;

      srcOpCanoniclizedMap.insert({sourceOp, {writeTensor, writeOpresult}});
      groupOpInitArgs[sourceOpGid].insert(writeTensor);
      groupOpResults[sourceOpGid].insert(
          {writeOpresult, {rtKind, OpAnchorPos[sourceOp]}});
      return;
    }
    generateEmptyTensorAndWrite(sourceOp, srcOpCanoniclizedMap,
                                OpAnchorPos[sourceOp], rtKind,
                                visitedOperation);
    return;
  }
  // udpate result return type
  updateReturnResultKind(srcOpCanoniclizedMap[sourceOp].second.getDefiningOp(),
                         sourceOpGid, rtKind);
}

void GroupOperationFusionImpl::GroupOperationReturnResultProcess(
    size_t sourceOpGid, Operation *sourceOp, Operation *op, size_t operandIdx,
    bool inSameGroupNeedReturn) {
  ReturnTypeKind rtKind = inSameGroupNeedReturn ? ReturnTypeKind::RT_InGroup
                                                : ReturnTypeKind::RT_OutGroup;
  SmallVector<SetVector<Value>, 8> &groupOpInitArgs =
      getGroupOperationFusion().getGroupOpInitArgs();

  DenseMap<Operation *, size_t> &opGroupIndexMap =
      getGroupOperationFusion().getOpGroupIndexMap();
  // update init iterargs
  auto dstRet = getOperationOperateTensor(sourceOp);
  // need to generate tensor.emtpy and vector.transfer_write, write
  // operand to tensor and read operand from the tensor, generate
  // vector.transfer_read
  if (failed(dstRet)) {
    // already generate result tensor, special operation do the
    // transformation by itself
    if (isSpecialOp(sourceOp) and inSameGroupNeedReturn and
        not isBroadcastOp(sourceOp))
      return;
    makeSourceOpWriteResultToTensor(sourceOp, sourceOpGid, rtKind);
    auto opInit = canonicalizeCurrentOperation(
        op, srcOpCanoniclizedMap[sourceOp].second, operandIdx);
    updateOpOperandResultInGroups(opGroupIndexMap[op], op, opInit);
    return;
  }
  // if source operation is transfer_read, we need to generate a
  // same transfer_read operation like source operation.
  if (isa<vector::TransferReadOp>(sourceOp)) {
    auto transferReadOp = cast<vector::TransferReadOp>(sourceOp);
    auto opInit = canonicalizeCurrentOperation(op, dstRet.value(), operandIdx,
                                               &transferReadOp);
    updateOpOperandResultInGroups(opGroupIndexMap[op], op, opInit);
    return;
  }
  // transfer write operation
  groupOpInitArgs[sourceOpGid].insert(dstRet.value());
  auto writeTensor = sourceOp->getOperand(1);
  if (dstRet.value() != writeTensor)
    getGroupOperationFusion().getOperandOriginalValue()[writeTensor] =
        dstRet.value();

  updateReturnResultKind(sourceOp, sourceOpGid, rtKind);
}

void GroupOperationFusionImpl::broadcastFromElements(Operation *op,
                                                     size_t grpIdx) {
  if (not isa<vector::BroadcastOp>(op))
    llvm_unreachable("Must be broadcast operation.");

  if (not isa<VectorType>(op->getOperandTypes()[0])) {
    auto inputBcastOp = cast<vector::BroadcastOp>(op);
    size_t steps = getGroupOperationFusion().getGroupMaxSteps()[grpIdx];
    IRRewriter rewriter(op);
    VectorType newOperandType =
        getGroupOperationFusion().getTypeHelper().getVectorzedType(op, steps);
    if (isa_and_nonnull<arith::ConstantOp>(op->getOperand(0).getDefiningOp())) {
      auto constantOp = cast<arith::ConstantOp>(op);
      SmallVector<int64_t> shapes(1, steps);
      auto dataType = mlir::VectorType::get(
          shapes, inputBcastOp.getResultVectorType().getElementType());

      FailureOr<Value> res = createArithSplatConstantOp(
          rewriter, op->getLoc(),
          DenseElementsAttr::get(dataType, constantOp.getValue()),
          newOperandType);
      if (failed(res))
        llvm::llvm_unreachable_internal("Wrong to create constant op.");
      removeOpInCurrentGroups(grpIdx, op, res.value().getDefiningOp());

    } else {
      auto bcastOp = rewriter.create<vector::BroadcastOp>(
          op->getLoc(), newOperandType, op->getOperands()[0]);
      removeOpInCurrentGroups(grpIdx, op, bcastOp);
      std::function<bool(Operation *)> candidateFunc = isBroadcastOp;
      moveSomeInterferenceOperation(&getGroupOperationFusion().getFunction(),
                                    op->getContext(), candidateFunc);
    }
  }
}

void GroupOperationFusionImpl::scalarOperandFromElements() {
  auto &opGroups = getGroupOperationFusion().getOpGroups();

  for (auto [idx, grp] : llvm::enumerate(opGroups)) {

    std::queue<Operation *> tmpQueue(grp);
    while (!tmpQueue.empty()) {
      auto op = tmpQueue.front();
      tmpQueue.pop();
      TypeSwitch<Operation *, void>(op)
          .Case<vector::BroadcastOp>([&](vector::BroadcastOp &bcOp) {
            broadcastFromElements(bcOp, idx);
          })
          .Default([&](Operation *op) { return; });
    }
  }
}

void GroupOperationFusionImpl::canonicalizeEachOperationGroup() {
  // record the operation which has been moved
  DenseSet<Operation *> movedOperationSet;
  //  record the operation's visited order, inorder to ensure set
  //  correct operand
  size_t opCounter = 0;
  DenseMap<Operation *, size_t> &opGroupIndexMap =
      getGroupOperationFusion().getOpGroupIndexMap();
  DenseMap<Operation *, size_t> &OpAnchorPos =
      getGroupOperationFusion().getOpAnchorPos();
  func::FuncOp func = getGroupOperationFusion().getFunction();
  IRRewriter rewriter(func);

  analysisGroupMaxSteps();

  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    visitedOperation.insert({op, opCounter++});

    for (auto [idx, opd] : llvm::enumerate(op->getOperands())) {
      Operation *sourceOp = opd.getDefiningOp();
      if (opGroupIndexMap.contains(sourceOp)) {
        auto sourceOpGid = opGroupIndexMap[sourceOp];
        bool notInSameGroup =
            opGroupIndexMap.contains(op) && sourceOpGid != opGroupIndexMap[op];
        bool outOfGroup = !opGroupIndexMap.contains(op);
        // Different anchor in same group and source operation is in inner
        // loop, we need to get source operation's result
        bool inSameGroupNeedReturn = !outOfGroup and !notInSameGroup and
                                     OpAnchorPos[sourceOp] > OpAnchorPos[op];

        if (notInSameGroup or outOfGroup or inSameGroupNeedReturn)
          GroupOperationReturnResultProcess(sourceOpGid, sourceOp, op, idx,
                                            inSameGroupNeedReturn);

        continue;
      }
      if (isa_and_nonnull<arith::ConstantOp>(sourceOp))
        replaceConstantOpAsNewOp(op, sourceOp, idx);
    }
  });
  analysisEmptyGroup();
  scalarOperandFromElements();
  specialOperationRectify(visitedOperation);
  LDBG("Complete analysis group operation results\n");
}

void ForLoopGenerator::rectifyGroupOperands(size_t currentGroupId,
                                            Value originalResult,
                                            Value forResult) {
  size_t totalGroupSize = getVectorBasedFusion().getOpGroups().size();
  size_t startGroup = currentGroupId;
  DenseMap<Value, Value> &operandOriginalMap =
      getVectorBasedFusion().getOperandOriginalValue();
  if (operandOriginalMap.contains(originalResult))
    originalResult = operandOriginalMap[originalResult];
  while (startGroup < totalGroupSize) {
    SetVector<Value> &operandVector =
        getVectorBasedFusion().getGroupOpInitArgs()[startGroup++];
    if (not operandVector.contains(originalResult))
      continue;
    SetVector<Value> replacedVector;

    for (auto v : operandVector) {
      if (v == originalResult) {
        replacedVector.insert(forResult);
        continue;
      }
      replacedVector.insert(v);
    }
    getVectorBasedFusion().getGroupOpInitArgs()[startGroup - 1] =
        replacedVector;
  }
}

mlir::FailureOr<scf::ForOp> ForLoopGenerator::generateVectorizedForLoop(
    const size_t groupId, IRRewriter &rewriter, VectorType vectorType) {
  // prepare for loop iterargs
  GenerateLoopHelper loopHelper(groupId, 0);
  prepareForLoopArgs(groupId, loopHelper);

  ArrayRef<int64_t> shapes = vectorType.getShape();
  // generate for loop
  auto forOp = constructNestedForOp(groupId, rewriter, rewriter.getUnknownLoc(),
                                    shapes, loopHelper);
  replaceOpUsersWithForLoopResult(forOp, groupId, loopHelper.nextAnchorResults,
                                  loopHelper.nextAnchorResultsIdxMap,
                                  loopHelper.nextAnchorResultOrignalResultMap);

  return forOp;
}

bool LoopGeneratorImpl::isGroupHasSpecialOperation(const size_t grpIdx) {
  auto &rdCanonicalizer = getMultiRdCanonicalizers()[grpIdx];
  auto &tpCanonicalizer = getTransposeCanonicalizers()[grpIdx];
  auto &shapeCastCanonicalizer = getShapeCastCanonicalizers()[grpIdx];
  return !rdCanonicalizer.getCandidateOps().empty() or
         !tpCanonicalizer.getCandidateOps().empty() or
         !shapeCastCanonicalizer.getCandidateOps().empty();
}

void LoopGeneratorImpl::generateGroupOpVectorizedIR(const int idx) {
  auto &grp = getVectorBasedFusion().getOpGroups()[idx];
  if (grp.empty()) {
    LDBG("Current operation Group is empty.");
    return;
  }
  // TODO: special operation better fusion
  if (isGroupHasSpecialOperation(idx))
    return;

  VectorType groupType =
      getVectorBasedFusion().getGroupBiggestRankVectorType()[idx];
  IRRewriter rewriter(grp.back());
  rewriter.setInsertionPointAfter(grp.back());
  // 1. Rewrite operation as vectorized form
  // 2. Generate loop
  // rewriteOperationAsVectorize(rewriter, idx);
  auto forOp = generateVectorizedForLoop(idx, rewriter, groupType);
  // special operation do not need to change anything
  if (failed(forOp))
    return;

  moveLoopInvariantCode(forOp.value());
}

/// Pass that lower to physical vector.
struct CPUPhysicalRegisterPass
    : public impl::CPUPhysicalRegisterPassBase<CPUPhysicalRegisterPass> {

  void runOnOperation() final {
    //
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    auto func = getOperation();

    if (hasNotSupportOperation(&func)) {
      LDBG("Not support operation appears in current function.");
      return;
    }
    // affineApply operation is always used by other operations.
    std::function<bool(Operation *)> candidateFunc = isProducerOp;
    moveSomeInterferenceOperation(&func, ctx, candidateFunc);
    candidateFunc = isCandidateMoveOperations;
    moveSomeInterferenceOperation(&func, ctx, candidateFunc);
    // canonicalize vector operation, default use vector-based fusion
    // strategy.
    HardWareInfo hwInfo;
    CPUTargetDescriptionAnalysis sysDesc =
        getAnalysis<CPUTargetDescriptionAnalysis>();
    hwInfo.favx512f = sysDesc.getMaxVectorWidth() >= 512;
    hwInfo.favx2 = sysDesc.getMaxVectorWidth() >= 256;
    VectorOperationCanonicalizer canonicalizer(
        func, hwInfo, CanonicalizerKind::GroupOperations);
    canonicalizer.run();

    candidateFunc = isReadOrWriteOperation;
    moveSomeInterferenceOperation(&func, ctx, candidateFunc);

    // transpose kernel
    vector::VectorTransformsOptions transposeOptions =
        vector::VectorTransformsOptions();
    transposeOptions.vectorTransposeLowering =
        vector::VectorTransposeLowering::Shuffle16x16;
    vector::populateVectorTransposeLoweringPatterns(patterns, transposeOptions);

    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace gc
} // namespace mlir