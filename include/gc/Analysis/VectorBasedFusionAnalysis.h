//===-- VectorBasedFusionAnalysis.h - vector fusion analysis ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_VECTORBASEDFUSIONANALYSIS_H
#define MLIR_ANALYSIS_VECTORBASEDFUSIONANALYSIS_H

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Dialect/Microkernel/MicrokernelOps.h"
#include "gc/Transforms/Passes.h"
#include "gc/Transforms/Utils/VectorUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include <queue>

namespace mlir {
namespace gc {

/// record hardware information
struct HardWareInfo {
  bool favx512f = true;
  bool favx2 = true;
};

/// Vector type conversion helper class
class TypeHelper {
private:
  HardWareInfo info;

public:
  TypeHelper() = default;
  TypeHelper(HardWareInfo info) : info(info) {}
  /// get current hardware information
  HardWareInfo &getHardwareInfo() { return this->info; }
  /// use \param info to set hardware information
  void setHardWareInfo(HardWareInfo &info) { this->info = info; }
  /// get vector \param type max loop step according to hardware information
  int getDataTypeValidSteps(VectorType type);
  /// get vector \param type an even for loop step
  int generateValidSteps(int steps, VectorType type);
  /// get vector \param type max simd length according to hardware information
  int getDataTypeMAXSIMDLength(VectorType type);
  /// get operation's vector type
  VectorType getVectorzedType(Operation *op, uint32_t loopStep = 0);
};

/// operation return kind, which is used to determine whether the operation
/// need to return it's result in current for loop
enum class ReturnTypeKind {
  RT_Both,
  RT_OutGroup,
  RT_InGroup,
};

class VectorFusionBase {

private:
  /// current function IR
  func::FuncOp func;
  /// Type helper class, can help us to get operation type
  TypeHelper typehelper;

public:
  VectorFusionBase() = default;
  VectorFusionBase(func::FuncOp &func, HardWareInfo &info)
      : func(func), typehelper(info) {}
  VectorFusionBase(VectorFusionBase &base)
      : func(base.getFunction()), typehelper(base.getHardwareInfo()) {}

  /// get current function IR
  func::FuncOp &getFunction() { return func; }
  /// get current hardware info
  HardWareInfo &getHardwareInfo() { return typehelper.getHardwareInfo(); }
  TypeHelper &getTypeHelper() { return typehelper; }
};

/// Group operation fusion strategy class.
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
class GroupOperationFusion : public VectorFusionBase {
private:
  /// operation groups, operations in each group can generate a common for
  /// loop
  SmallVector<std::queue<Operation *>, 8> opGroups;
  /// group max vectorize steps
  SmallVector<uint32_t, 8> groupMaxSteps;
  /// vector type which has bigest rank in current operation group
  llvm::SmallDenseMap<size_t, VectorType> groupBigestRankVectorType;
  /// query current operation in which group, return group index
  DenseMap<Operation *, size_t> opGroupIndexMap;
  /// can fused into prev operation which axis position
  DenseMap<Operation *, size_t> opAnchorPos;
  /// record some operations which not need to No need to judge whether can be
  /// fused
  std::queue<Operation *> notNeedToJudgeOps;
  /// analysis the operation's operands and results
  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
      groupOpResults;
  /// store loop iteration args for each of operation group
  SmallVector<SetVector<Value>, 8> groupOpInitArgs;
  // store read and write operations permutation maps in order to convenient
  // to replace loop induction var
  DenseMap<Operation *, AffineMap> opPermuationMap;
  /// record operation operand original operate value
  DenseMap<Value, Value> operandOriginalValue;

public:
  GroupOperationFusion(func::FuncOp &func, HardWareInfo &info)
      : VectorFusionBase(func, info) {}

  GroupOperationFusion(GroupOperationFusion &strategy)
      : VectorFusionBase(strategy.getFunction(), strategy.getHardwareInfo()),
        opGroups(strategy.opGroups), groupMaxSteps(strategy.groupMaxSteps),
        opGroupIndexMap(strategy.opGroupIndexMap),
        opAnchorPos(strategy.opAnchorPos){};

  GroupOperationFusion(GroupOperationFusion &&strategy)
      : VectorFusionBase(strategy.getFunction(), strategy.getHardwareInfo()),
        opGroups(std::move(strategy.opGroups)),
        groupMaxSteps(std::move(strategy.groupMaxSteps)),
        groupBigestRankVectorType(
            std::move(strategy.getGroupBiggestRankVectorType())),
        opGroupIndexMap(std::move(strategy.opGroupIndexMap)),
        opAnchorPos(std::move(strategy.opAnchorPos)){};

  GroupOperationFusion &operator=(GroupOperationFusion &fusion) {
    this->getOpGroups() = fusion.getOpGroups();
    this->getGroupMaxSteps() = fusion.getGroupMaxSteps();
    this->getGroupBiggestRankVectorType() =
        fusion.getGroupBiggestRankVectorType();
    this->getOpGroupIndexMap() = fusion.getOpGroupIndexMap();
    this->getOpAnchorPos() = fusion.getOpAnchorPos();
    this->notNeedToJudgeOps = fusion.notNeedToJudgeOps;
    this->getGroupOpResults() = fusion.getGroupOpResults();
    this->getGroupOpInitArgs() = fusion.getGroupOpInitArgs();
    this->getOpPermuationMap() = fusion.getOpPermuationMap();
    this->getOperandOriginalValue() = fusion.getOperandOriginalValue();
    this->getFunction() = fusion.getFunction();
    this->getHardwareInfo() = fusion.getHardwareInfo();
    this->getTypeHelper() = fusion.getTypeHelper();
    return *this;
  };
  GroupOperationFusion &operator=(GroupOperationFusion &&) = default;

  /// Get the map which contains each group vector type which has biggest
  /// rank.
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
  SmallVector<uint32_t, 8> &getGroupMaxSteps() noexcept {
    return groupMaxSteps;
  }
  /// Get the map contains anchor position of each operation
  DenseMap<Operation *, size_t> &getOpAnchorPos() noexcept {
    return opAnchorPos;
  }
  /// get current operation group results
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

  DenseMap<Value, Value> &getOperandOriginalValue() noexcept {
    return operandOriginalValue;
  }
  /// set operation groups
  void setGroupOpResults(
      const SmallVector<
          llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
          &results) {
    groupOpResults = std::move(results);
  }

  void setGroupOpIterArgs(
      const SmallVector<llvm::SetVector<Value>, 8> &initArgs) noexcept {
    groupOpInitArgs = std::move(initArgs);
  }

  void setPermutationMap(const DenseMap<Operation *, AffineMap> &map) noexcept {
    opPermuationMap = std::move(map);
  }
  /// Do fusion strategy
  void classifyOperations();

  /// Whether two operations have compatible vector shapes
  bool isCompatibleVectorType(Operation *op1, Operation *op2);

  /// update bigest vector type for last operation group
  void updateGroupBigestVectorType(VectorType vectorType);

  /// Check whether the operation can fuse with previous operation
  bool isNeedNewGroup(Operation *op);

  /// Add Operation \p op into current last group or a new Group
  /// \p op must has valid value, can't be nullptr
  void addOperationToGroup(Operation *op);

  /// get next operation in current operation group
  template <typename Target>
  Operation *getNextTargetOperationInCurrentGroup(Operation *curOp,
                                                  const size_t grpIdx);

  /// run the vector-based fusion strategy
  void run();
};

template <typename Target>
Operation *GroupOperationFusion::getNextTargetOperationInCurrentGroup(
    Operation *curOp, const size_t grpIdx) {
  std::queue<Operation *> tmpOpQueue(getOpGroups()[grpIdx]);
  if (isa<Target>(curOp))
    return curOp;

  while (!tmpOpQueue.empty()) {
    auto frontOp = tmpOpQueue.front();
    if (isa<Target>(frontOp)) {
      for (auto x : frontOp->getOperands())
        if (x.getDefiningOp() == curOp)
          return frontOp;
    }
    tmpOpQueue.pop();
  }
  return nullptr;
}

class GroupOperationAnalysis {
private:
  /// vector-based fusion related data
  GroupOperationFusion fusionStrategy;

public:
  GroupOperationAnalysis(func::FuncOp &func, HardWareInfo &info)
      : fusionStrategy(func, info) {}
  /// remove the useless operation, due to it result is not require by other
  /// operation
  void analysisEmptyGroup();
  /// get each operation in each group maximum support vectorization length
  void analysisGroupMaxSteps();
  /// get fusion strategy
  GroupOperationFusion &getGroupOperationFusion() { return fusionStrategy; }

  void run() { fusionStrategy.run(); }
};
} // namespace gc
} // namespace mlir

#endif