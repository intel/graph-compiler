//===-- ConstantTensorFolding.cpp - Constant Folding ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs a constant subgraph transform in MLIR.
//
//===----------------------------------------------------------------------===//
#include <deque>
#include <iostream>
#include <unordered_set>

#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "gc/ExecutionEngine/CPURuntime/ConstantCache.h"

#define DEBUG_TYPE "constant-tensor-folding"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CONSTANTTENSORFOLDING
#include "gc/Transforms/Passes.h.inc"
} // namespace gc

using namespace mlir;

namespace gc {

struct ConstantTensorFolding
    : public impl::ConstantTensorFoldingBase<ConstantTensorFolding> {
  void runOnOperation() override;
};

bool isInConstantSubgraph(Operation *op) {
  auto opNamespace = op->getDialect()->getNamespace();
  if (opNamespace == linalg::LinalgDialect::getDialectNamespace() ||
      opNamespace == tensor::TensorDialect::getDialectNamespace() ||
      opNamespace ==
          bufferization::BufferizationDialect::getDialectNamespace() ||
      opNamespace == arith::ArithDialect::getDialectNamespace()) {
    if (op->getAttr("onednn_graph.in_const_subgraph")) {
      return true;
    }
  }
  return false;
}

template <typename T> int64_t getDataSize(T t) {
  Type eleType = t.getElementType();
  unsigned bitWidth = eleType.getIntOrFloatBitWidth() / 8; // bytes
  ArrayRef<int64_t> shape = t.getShape();
  int64_t size = bitWidth;
  for (auto s : shape)
    size *= s;

  return size;
}

int64_t getValueSize(Value v) {
  if (isa<TensorType>(v.getType())) {
    auto t = dyn_cast<TensorType>(v.getType());
    return getDataSize<TensorType>(t);
  } else {
    auto t = dyn_cast<MemRefType>(v.getType());
    return getDataSize<MemRefType>(t);
  }
}

/// @brief op has only one operand, or operands of op are one same value, or
/// operands of op are one same value or from tensor.EmptyOp.
/// @param op
/// @return
bool singleOperand(Operation *op) {
  if (op->getNumOperands() > 1) {
    Value firstOperand = op->getOperand(0);
    for (int64_t i = 1; i < op->getNumOperands(); ++i) {
      Value operand = op->getOperand(i);
      if (firstOperand == operand)
        continue;

      auto parentOp = operand.getDefiningOp();
      if (parentOp && !isa<tensor::EmptyOp>(parentOp))
        return false;
    }
  }
  return true;
}

bool canMoveBefore(Operation *op) {
  if (op->getDialect()->getNamespace() ==
      arith::ArithDialect::getDialectNamespace()) {
    return true;
  }

  if (op->getDialect()->getNamespace() !=
      linalg::LinalgDialect::getDialectNamespace()) {
    return false;
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);

  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  for (auto &affineMap : indexingMaps) {
    if (!affineMap.isIdentity())
      return false;
  }

  SmallVector<utils::IteratorType> iterTypes = linalgOp.getIteratorTypesArray();
  for (auto &iterType : iterTypes) {
    if (iterType != utils::IteratorType::parallel)
      return false;
  }

  if (op->getNumOperands() > 1) {
    // int64_t numInputs = linalgOp.getNumDpsInputs();
    int64_t numInits = linalgOp.getNumDpsInits();
    // definingOp of init should be tensor.empty()
    for (int64_t i = 0; i < numInits; ++i) {
      OpOperand *outOperand = linalgOp.getDpsInitOperand(i);
      auto parentOp = outOperand->get().getDefiningOp();
      if (!isa<tensor::EmptyOp>(parentOp))
        return false;
    }
  }

  return true;
}

void postponeBroadcast(Block &block) {
  // auto bcOps = block.getOps<linalg::BroadcastOp>();
  // for (linalg::BroadcastOp bcOp : bcOps) {}
  SmallVector<Operation *> constBcOps;
  for (Operation &op : block.getOperations()) {
    if (isa<linalg::BroadcastOp>(&op)) {
      Operation *bcOp = &op;
      if (isInConstantSubgraph(bcOp))
        constBcOps.push_back(bcOp);
    }
  }

  for (auto bcOp : constBcOps) {
    // For topo v -> pack -> bc -> mul -> matmul, we transform
    // it to v -> pack -> mul -> bc -> matmul, so that we can fold
    // v -> pack -> mul. Note that we require the topo to be sequential
    // and all the Values have exactly one user.

    // go upwards to BlockArg
    SmallVector<Operation *> prevOps;
    Operation *currOp = bcOp;
    while (true) {
      if (currOp->getNumOperands() != 1)
        break;

      Value operand = currOp->getOperand(0);
      if (isa<BlockArgument>(operand)) {
        break;
      } else {
        currOp = operand.getDefiningOp();
        prevOps.push_back(currOp);
      }
    }

    // go downwards to the last constant op
    SmallVector<Operation *> postOps;
    currOp = bcOp;
    while (true) {
      if (currOp->getNumResults() != 1 || !currOp->hasOneUse())
        break;

      Value input = currOp->getResult(0);
      currOp = *(input.getUsers().begin());
      Value output = currOp->getResult(0);
      // NOTE: we require that input shape and output shape of curr op to be
      // same. Operations from tensor dialect, like
      // pack/unpack/concat/collapse_shape/expand_shape/reshape/pad, are not
      // supported. So we simply restrict that currOp to be from arith or
      // linalg.
      if (!isa<TensorType>(input.getType()) ||
          !isa<TensorType>(output.getType()) ||
          dyn_cast<TensorType>(input.getType()).getShape() !=
              dyn_cast<TensorType>(output.getType()).getShape() ||
          !canMoveBefore(currOp)) {
        break;
      }
      if (!isInConstantSubgraph(currOp)) {
        break;
      } else {
        postOps.push_back(currOp);
      }
    }
    if (postOps.empty())
      continue;

    // move bcOp after the last constant op
    SmallVector<Operation *> newPostOps;
    Value operand = static_cast<Value>(bcOp->getOperand(0));
    ArrayRef<int64_t> shapeBeforeBc =
        dyn_cast<TensorType>(operand.getType()).getShape();
    size_t postOpId = 0;
    for (Operation *postOp : postOps) {
      SmallVector<Type> newOperandTypes;
      for (auto oriType : postOp->getOperandTypes()) {
        TensorType tt = dyn_cast<TensorType>(oriType);
        newOperandTypes.push_back(
            tt.cloneWith(shapeBeforeBc, tt.getElementType()));
      }
      SmallVector<Type> newResultTypes;
      for (auto oriType : postOp->getResultTypes()) {
        TensorType tt = dyn_cast<TensorType>(oriType);
        newResultTypes.push_back(
            tt.cloneWith(shapeBeforeBc, tt.getElementType()));
      }
      auto *newPostOp =
          Operation::create(postOp->getLoc(), postOp->getName(), newResultTypes,
                            postOp->getOperands(),
                            /*postOp->getAttrDictionary()*/ std::nullopt,
                            /*postOp->getPropertiesStorage()*/ nullptr,
                            postOp->getSuccessors(), postOp->getNumRegions());
      for (auto [oldRegion, newRegion] :
           llvm::zip(postOp->getRegions(), newPostOp->getRegions())) {
        newRegion.takeBody(oldRegion);
      }

      if (postOpId == 0) {
        // Only the first post op needs to replace its operand. Others only
        // needs to call postOp->replaceAllUsesWith(newPostOp->getResults()).
        newPostOp->getOperand(0).replaceAllUsesWith(operand);
      }
      ++postOpId;

      newPostOp->setAttr("onednn_graph.in_const_subgraph",
                         postOp->getAttr("onednn_graph.in_const_subgraph"));
      if (postOp->getDialect()->getNamespace() ==
          linalg::LinalgDialect::getDialectNamespace()) {
        newPostOp->setAttr("operandSegmentSizes",
                           postOp->getAttr("operandSegmentSizes"));

        OpBuilder builder(postOp->getContext());
        size_t indexingMapsSize =
            dyn_cast<linalg::LinalgOp>(postOp).getIndexingMapsArray().size();
        unsigned rank = shapeBeforeBc.size();
        SmallVector<AffineMap> indexingMaps(
            indexingMapsSize, builder.getMultiDimIdentityMap(rank));
        auto indexingMapsAttr = builder.getAffineMapArrayAttr(indexingMaps);
        newPostOp->setAttr("indexing_maps", indexingMapsAttr);

        SmallVector<utils::IteratorType> iterTypes =
            dyn_cast<linalg::LinalgOp>(postOp).getIteratorTypesArray();
        iterTypes.resize(rank);
        auto iterTypesAttr =
            builder.getArrayAttr(llvm::to_vector(llvm::map_range(
                iterTypes, [&](utils::IteratorType iter) -> mlir::Attribute {
                  return linalg::IteratorTypeAttr::get(builder.getContext(),
                                                       iter);
                })));
        newPostOp->setAttr("iterator_types", iterTypesAttr);
      } else {
        // Ops from other dialects.
      }

      // Modify the outputOperands of postOp. Here we simply assume that the
      // value is from tensor.empty().
      if (postOp->getNumOperands() > 0) {
        for (size_t i = 1; i < postOp->getNumOperands(); ++i) {
          auto outOperand = postOp->getOperand(i);
          outOperand.setType(newOperandTypes.front());
        }
      }

      block.getOperations().push_back(newPostOp);
      newPostOp->moveAfter(postOp);
      newPostOps.push_back(newPostOp);
      postOp->replaceAllUsesWith(newPostOp->getResults());

      operand = static_cast<Value>(newPostOp->getResult(0));
    }

    auto nextOp = *(newPostOps.back()->getUsers().begin());
    nextOp->getOperand(0).replaceAllUsesWith(bcOp->getResult(0));
    bcOp->moveAfter(newPostOps.back());
    bcOp->getOperand(0).replaceUsesWithIf(operand, [&](OpOperand &val) {
      Operation *op = val.getOwner();
      return op == bcOp;
    });

    for (auto it = postOps.rbegin(); it != postOps.rend(); ++it)
      (*it)->erase();
  }
}

static void addGlobalI32(ModuleOp &module, Location loc, OpBuilder &builder,
                         StringRef name, int32_t value) {
  OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto type = IntegerType::get(builder.getContext(), 32);
  LLVM::GlobalOp global = builder.create<LLVM::GlobalOp>(
      loc, type, /*isConstant=*/true, LLVM::Linkage::External, name,
      builder.getI32IntegerAttr(value),
      /*alignment=*/0);
  (void)global;
}

static void addGlobalI64Array(ModuleOp &module, Location loc,
                              OpBuilder &builder, StringRef name,
                              ArrayRef<int64_t> array) {
  OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto type = LLVM::LLVMArrayType::get(
      IntegerType::get(builder.getContext(), 64), array.size());
  LLVM::GlobalOp global = builder.create<LLVM::GlobalOp>(
      loc, type, /*isConstant=*/true, LLVM::Linkage::External, name,
      builder.getI64TensorAttr(array),
      /*alignment=*/0);
  (void)global;
}

static void addGlobalI32Array(ModuleOp &module, Location loc,
                              OpBuilder &builder, StringRef name,
                              ArrayRef<int32_t> array) {
  OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto type = LLVM::LLVMArrayType::get(
      IntegerType::get(builder.getContext(), 32), array.size());
  LLVM::GlobalOp global = builder.create<LLVM::GlobalOp>(
      loc, type, /*isConstant=*/true, LLVM::Linkage::External, name,
      builder.getI32TensorAttr(array),
      /*alignment=*/0);
  (void)global;
}

std::unordered_set<int> getConstArgsIndexes(Operation &topFunc,
                                            bool compiletime) {
  auto topFuncAttr = topFunc.getAttrDictionary();
  std::unordered_set<int> constArgsIndexes;
  std::string attrName =
      compiletime ? "compiletime_const_args_index" : "runtime_const_args_index";
  std::optional<NamedAttribute> constArgs = topFuncAttr.getNamed(attrName);
  if (constArgs.has_value()) {
    for (auto id : llvm::dyn_cast<ArrayAttr>(constArgs->getValue())) {
      constArgsIndexes.insert(llvm::cast<IntegerAttr>(id).getInt());
    }
  }
  return constArgsIndexes;
}

void getArithConstantOutputs(Block &block, SmallVector<Type> &outputTypes,
                             SmallVector<Value> &outputValues) {
  for (Operation &op : block.getOperations()) {
    if (isa<arith::ConstantOp>(&op)) {
      Operation *constOp = &op;
      auto constTensor = constOp->getResults().front();
      if (!isa<TensorType>(constTensor.getType()))
        continue;

      auto v = dyn_cast<Value>(constTensor);
      SmallVector<Value> valuesOnTheWay = {v}; // the constant tensors
      std::deque<Value> dq;
      dq.push_back(v);
      // For v -> pack1 -> pack2 -> matmul, we need the type of output of pack2
      while (!dq.empty()) {
        v = dq.front();
        dq.pop_front();
        // if the children ops of v are not all constant, we end at v
        if (std::any_of(v.getUsers().begin(), v.getUsers().end(),
                        [](Operation *child) {
                          return !isInConstantSubgraph(child);
                        })) {
          if (std::find(outputValues.begin(), outputValues.end(), v) ==
              outputValues.end()) {
            outputTypes.push_back(v.getType());
            outputValues.push_back(v);
          }
          continue;
        }

        // the children ops of v are all constant, we push their results to
        // queue
        for (Operation *child : v.getUsers()) {
          for (OpResult result : child->getResults()) {
            auto r = dyn_cast<Value>(result);
            dq.push_back(r);
            valuesOnTheWay.push_back(r);
          }
        }
      }
    }
  }
}

static constexpr int DATA_SIZE_EXPANDING_THRESHOLD = 8;

void getInputsAndOutputs(Block &block,
                         std::unordered_set<int> &constArgsIndexes,
                         SmallVector<Type> &inputTypes,
                         SmallVector<Value> &inputValues,
                         SmallVector<Type> &outputTypes,
                         SmallVector<Value> &outputValues) {
  Value v;
  // Support complicated topology.
  for (size_t id = 0; id < block.getNumArguments(); ++id) {
    if (constArgsIndexes.count(id) == 1) {
      // The constant ops are all single-input single-output.
      bool simpleTopo = true;
      auto arg = block.getArgument(id);
      if (!isa<TensorType>(arg.getType()) && !isa<MemRefType>(arg.getType())) {
        continue;
      }
      inputTypes.push_back(arg.getType());
      v = dyn_cast<Value>(arg);
      inputValues.push_back(v);
      SmallVector<Value> valuesOnTheWay = {v}; // the constant tensors
      std::deque<Value> dq;
      dq.push_back(v);
      // For v -> pack1 -> pack2 -> matmul, we need the type of output of pack2
      while (!dq.empty()) {
        v = dq.front();
        dq.pop_front();
        // if the children ops of v are not all constant, we end at v
        if (std::any_of(v.getUsers().begin(), v.getUsers().end(),
                        [](Operation *child) {
                          return !isInConstantSubgraph(child);
                        })) {
          // skip case: memref v -> bufferization.to_tensor -> tensor t.
          if (valuesOnTheWay.size() == 2 && v.hasOneUse() &&
              isa<bufferization::ToTensorOp>(v.getDefiningOp())) {
            inputTypes.pop_back();
            inputValues.pop_back();
            constArgsIndexes.erase(id);
            continue;
          }
          if (std::find(outputValues.begin(), outputValues.end(), v) ==
              outputValues.end()) {
            outputTypes.push_back(v.getType());
            outputValues.push_back(v);
          }
          continue;
        }
        if (!v.hasOneUse())
          simpleTopo = false;

        // the children ops of v are all constant, we push their results to
        // queue
        for (Operation *child : v.getUsers()) {
          if (!singleOperand(child) || child->getResults().size() > 1)
            simpleTopo = false;

          for (OpResult result : child->getResults()) {
            auto r = dyn_cast<Value>(result);
            dq.push_back(r);
            valuesOnTheWay.push_back(r);
          }
        }
      }

      // If data size of outputValue is too greater than size of inputValue, do
      // not fold it. Compare data size changes during traverse to find the last
      // op that satisfies this condition.
      if (simpleTopo) {
        int64_t initSize = getValueSize(valuesOnTheWay[0]);
        if (initSize * DATA_SIZE_EXPANDING_THRESHOLD <
            getValueSize(valuesOnTheWay.back())) {
          size_t lastIdx = 0;
          for (size_t i = 1; i < valuesOnTheWay.size(); ++i) {
            int64_t size = getValueSize(valuesOnTheWay[i]);
            if (initSize * DATA_SIZE_EXPANDING_THRESHOLD > size) {
              lastIdx = i;
            }
          }
          if (lastIdx == 0) { // no suitable value found
            inputTypes.pop_back();
            outputTypes.pop_back();
            inputValues.pop_back();
            outputValues.pop_back();
            constArgsIndexes.erase(id);
          } else {
            outputTypes.back() = valuesOnTheWay[lastIdx].getType();
            outputValues.back() = valuesOnTheWay[lastIdx];
          }
        }
      }
    }
  }
}

func::FuncOp buildFoldFunc(MLIRContext *context, OpBuilder &builder,
                           Operation *topOp, const std::string &name,
                           const SmallVector<Operation *> &constOps,
                           SmallVector<Type> &inputTypes,
                           SmallVector<Value> &inputValues,
                           SmallVector<Type> &outputTypes,
                           SmallVector<Value> &outputValues) {
  FunctionType foldFuncType =
      FunctionType::get(context, inputTypes, outputTypes);
  func::FuncOp foldFunc =
      builder.create<func::FuncOp>(topOp->getLoc(), name, foldFuncType);
  Block *foldBlock = foldFunc.addEntryBlock();
  // values of folded constant tensors in foldBlock
  SmallVector<Value> outputValuesInFold;
  IRMapping mapper;
  for (Operation *op : constOps) {
    foldBlock->getOperations().push_back(op->clone(mapper));
  }
  // the order of outputValuesInFold is according to the order of corresponding
  // inputValues
  for (auto &v : outputValues) {
    auto foldedV = mapper.lookupOrNull(v);
    outputValuesInFold.push_back(foldedV);
    v.replaceUsesWithIf(foldedV, [&](OpOperand &val) {
      Operation *op = val.getOwner();
      return op->getBlock() == foldBlock;
    });
  }

  // Allocate buffer for outputValuesInFold
  std::vector<size_t> buffersSize;
  for (Value &tensor : outputValuesInFold) {
    LLVM_DEBUG(llvm::dbgs()
               << "Allocate buffer for tensor: " << tensor << "\n");
    buffersSize.push_back(getValueSize(tensor));
  }
  auto cacheManager = ConstGraphTensorCacheManager::get();
  SmallVector<int64_t> globalIndexes;
  for (auto id : cacheManager->alloc(buffersSize))
    globalIndexes.push_back(id);

  globalIndexes.insert(globalIndexes.begin(), globalIndexes.size());
  auto moduleOp = dyn_cast<ModuleOp>(topOp);
  addGlobalI64Array(moduleOp, moduleOp.getLoc(), builder,
                    "__" + name + "_buffer_ids", globalIndexes);

  auto returnOp =
      builder.create<func::ReturnOp>(topOp->getLoc(), outputValuesInFold);
  foldBlock->getOperations().push_back(returnOp);
  for (size_t i = 0; i < inputValues.size(); ++i) {
    inputValues[i].replaceUsesWithIf(foldBlock->getArgument(i),
                                     [&](OpOperand &val) {
                                       Operation *op = val.getOwner();
                                       return op->getBlock() == foldBlock;
                                     });
  }

  // the ranks of folded results.
  SmallVector<int32_t> foldRanks;
  // the shapes of folded results.
  SmallVector<int64_t> foldShapes;
  for (Value &tensor : outputValuesInFold) {
    auto t = dyn_cast<TensorType>(tensor.getType());
    Type eleType = t.getElementType();
    int64_t bitWidth = eleType.getIntOrFloatBitWidth() / 8; // bytes
    ArrayRef<int64_t> shape = t.getShape();
    foldRanks.push_back(shape.size());
    foldShapes.insert(foldShapes.end(), shape.begin(), shape.end());
    foldShapes.push_back(bitWidth);
  }
  addGlobalI32Array(moduleOp, moduleOp.getLoc(), builder, "__folded_ranks",
                    foldRanks);
  addGlobalI64Array(moduleOp, moduleOp.getLoc(), builder, "__folded_shapes",
                    foldShapes);

  foldFunc.setVisibility(SymbolTable::Visibility::Public);
  foldFunc->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));

  moduleOp.push_back(foldFunc);
  SymbolTable symbolTable(moduleOp);
  symbolTable.insert(foldFunc);

  return foldFunc;
}

void modifyComputeFunc(MLIRContext *context, OpBuilder &builder,
                       Operation *topOp, Operation &func, Block &block,
                       std::unordered_set<int> &constArgsIndexes,
                       SmallVector<Type> &outputTypes,
                       SmallVector<Value> &outputValues) {
  // the indexes of args to the folding func, including to-fold tensors and
  // folded results.
  SmallVector<int32_t> foldArgs;
  // the indexes of folded results.
  SmallVector<int32_t> foldIds;
  // the indexes of args to the computing func, including non-fold tensors and
  // folded results.
  SmallVector<int32_t> computeArgs;

  // modify the BlockArguments of block
  size_t oriNumArgs = block.getNumArguments();
  // Add the folded args to the end of BlockArguments list
  for (size_t id = 0; id < outputValues.size(); ++id) {
    auto loc = block.getArgument(id).getLoc();
    BlockArgument foldArg =
        block.insertArgument(oriNumArgs + id, outputTypes[id], loc);
    outputValues[id].replaceUsesWithIf(foldArg, [&](OpOperand &val) {
      Operation *op = val.getOwner();
      return op->getBlock() == &block;
    });
    foldIds.push_back(id + oriNumArgs);
  }
  // Erase the operations on constant args
  for (size_t id = 0; id < oriNumArgs; ++id) {
    if (constArgsIndexes.count(id) == 1) {
      foldArgs.push_back(id);
      std::deque<Value> dq;
      SmallVector<Operation *> opsToErase;
      std::unordered_set<Operation *> opsToEraseSet;
      dq.push_back(block.getArgument(id));
      while (!dq.empty()) {
        Value v = dq.front();
        dq.pop_front();
        for (Operation *op : v.getUsers()) {
          for (auto res : op->getResults()) {
            dq.push_back(res);
          }
          if (opsToEraseSet.count(op)) {
            break;
          }
          opsToErase.push_back(op);
          opsToEraseSet.insert(op);
        }
      }
      for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it) {
        (*it)->erase();
      }
    } else {
      computeArgs.push_back(id);
    }
  }
  // Erase the constant args in BlockArguments list
  llvm::BitVector argsToErase;
  for (size_t id = 0; id < oriNumArgs; ++id) {
    if (constArgsIndexes.count(id) == 1) {
      argsToErase.push_back(true);
    } else {
      argsToErase.push_back(false);
    }
  }
  for (size_t id = 0; id < outputValues.size(); ++id) {
    argsToErase.push_back(false);
  }
  block.eraseArguments(argsToErase);

  // modify the compute func signature
  func::FuncOp computeFunc = cast<func::FuncOp>(func);
  FunctionType computeFuncType = computeFunc.getFunctionType();
  computeFunc.setType(FunctionType::get(context, block.getArgumentTypes(),
                                        computeFuncType.getResults()));

  auto moduleOp = dyn_cast<ModuleOp>(topOp);
  for (auto id : foldIds) {
    foldArgs.insert(foldArgs.end(), id);
  }
  foldArgs.insert(foldArgs.begin(), foldArgs.size());
  addGlobalI32Array(moduleOp, moduleOp.getLoc(), builder, "__fold_args",
                    foldArgs);

  for (auto id : foldIds) {
    computeArgs.insert(computeArgs.end(), id);
  }
  computeArgs.insert(computeArgs.begin(), computeArgs.size());
  addGlobalI32Array(moduleOp, moduleOp.getLoc(), builder, "__compute_args",
                    computeArgs);

  addGlobalI32(moduleOp, moduleOp.getLoc(), builder, "__num_orig_args",
               oriNumArgs);
}

void canonicalizeAndClean(MLIRContext *context, Operation *topOp) {
  // Delete dead operations by dialects' canonicalizer
  RewritePatternSet owningPatterns(context);
  for (auto *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(owningPatterns);

  ArrayRef<std::string> disabledPatterns, enabledPatterns;
  std::shared_ptr<const FrozenRewritePatternSet> patterns =
      std::make_shared<FrozenRewritePatternSet>(
          std::move(owningPatterns), disabledPatterns, enabledPatterns);
  GreedyRewriteConfig config;
  LogicalResult converged =
      applyPatternsAndFoldGreedily(topOp, *patterns, config);
  (void)converged;

  // clean up the constant-related attrs on ops
  topOp->walk([&](Operation *op) {
    if (op->getAttr("onednn_graph.in_const_subgraph")) {
      op->removeAttr("onednn_graph.in_const_subgraph");
    }
  });
  topOp->walk([&](func::FuncOp op) {
    if (op.getOperation()->getAttr("compiletime_const_args_index")) {
      op.getOperation()->removeAttr("compiletime_const_args_index");
    }
    if (op.getOperation()->getAttr("runtime_const_args_index")) {
      op.getOperation()->removeAttr("runtime_const_args_index");
    }
  });
}

// Operate on tensors. Create fold() and compute() on module. The
// folded weights and first-run flag is maintained by upper-level runtime.
void ConstantTensorFolding::runOnOperation() {
  Operation *topOp = getOperation();
  MLIRContext *context = topOp->getContext();
  auto &topFunc =
      topOp->getRegions().front().getBlocks().front().getOperations().front();
  OpBuilder builder(context);
  Region &region = topFunc.getRegions().front();
  Block &block = region.getBlocks().front();

  std::unordered_set<int> compiletimeConstArgsIndexes =
      getConstArgsIndexes(topFunc, true);
  std::unordered_set<int> runtimeConstArgsIndexes =
      getConstArgsIndexes(topFunc, false);
  if (compiletimeConstArgsIndexes.empty() && runtimeConstArgsIndexes.empty()) {
    return;
  }

  postponeBroadcast(block);

  SmallVector<Operation *> constOps;
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isInConstantSubgraph(&op)) {
      constOps.push_back(&op);
    }
  }

  bool enableCompiletimeFolding = false;
  if (enableCompiletimeFolding) {
    // ===== build compile time folding function =====
    SmallVector<Type> compiletimeInputTypes; // types of constant tensors
    // values of constant tensors in original block
    SmallVector<Value> compiletimeInputValues;
    SmallVector<Type>
        compiletimeOutputTypes; // types of folded constant tensors
    // values of folded constant tensors in original block
    SmallVector<Value> compiletimeOutputValues;
    getArithConstantOutputs(block, compiletimeOutputTypes,
                            compiletimeOutputValues);
    getInputsAndOutputs(block, compiletimeConstArgsIndexes,
                        compiletimeInputTypes, compiletimeInputValues,
                        compiletimeOutputTypes, compiletimeOutputValues);
    assert(compiletimeInputTypes.size() == compiletimeInputValues.size());
    assert(compiletimeOutputTypes.size() == compiletimeOutputValues.size());

    if (!compiletimeOutputTypes.empty()) {
      func::FuncOp compiletimeFoldFunc =
          buildFoldFunc(context, builder, topOp, "compiletime_fold", constOps,
                        compiletimeInputTypes, compiletimeInputValues,
                        compiletimeOutputTypes, compiletimeOutputValues);
      (void)compiletimeFoldFunc;
      canonicalizeAndClean(context, compiletimeFoldFunc.getOperation());
    }

    // ===== build runtime folding function =====
    SmallVector<Type> runtimeInputTypes; // types of constant tensors
    // values of constant tensors in original block
    SmallVector<Value> runtimeInputValues;
    SmallVector<Type> runtimeOutputTypes; // types of folded constant tensors
    // values of folded constant tensors in original block
    SmallVector<Value> runtimeOutputValues;
    getInputsAndOutputs(block, runtimeConstArgsIndexes, runtimeInputTypes,
                        runtimeInputValues, runtimeOutputTypes,
                        runtimeOutputValues);
    assert(runtimeInputTypes.size() == runtimeInputValues.size());
    assert(runtimeOutputTypes.size() == runtimeOutputValues.size());

    if (!runtimeOutputTypes.empty()) {
      func::FuncOp runtimeFoldFunc = buildFoldFunc(
          context, builder, topOp, "runtime_fold", constOps, runtimeInputTypes,
          runtimeInputValues, runtimeOutputTypes, runtimeOutputValues);
      (void)runtimeFoldFunc;
      canonicalizeAndClean(context, runtimeFoldFunc.getOperation());
    }

    // ===== build computing function =====
    std::unordered_set<int> constArgsIndexes = compiletimeConstArgsIndexes;
    constArgsIndexes.merge(runtimeConstArgsIndexes);
    SmallVector<Type> outputTypes = compiletimeOutputTypes;
    outputTypes.insert(outputTypes.end(), runtimeOutputTypes.begin(),
                       runtimeOutputTypes.end());
    SmallVector<Value> outputValues = compiletimeOutputValues;
    outputValues.insert(outputValues.end(), runtimeOutputValues.begin(),
                        runtimeOutputValues.end());
    if (!outputTypes.empty()) {
      modifyComputeFunc(context, builder, topOp, topFunc, block,
                        constArgsIndexes, outputTypes, outputValues);
    }
  } else {
    std::unordered_set<int> constArgsIndexes = compiletimeConstArgsIndexes;
    constArgsIndexes.merge(runtimeConstArgsIndexes);

    // ===== build runtime folding function =====
    SmallVector<Type> inputTypes; // types of constant tensors
    // values of constant tensors in original block
    SmallVector<Value> inputValues;
    SmallVector<Type> outputTypes; // types of folded constant tensors
    // values of folded constant tensors in original block
    SmallVector<Value> outputValues;
    getArithConstantOutputs(block, outputTypes, outputValues);
    getInputsAndOutputs(block, constArgsIndexes, inputTypes, inputValues,
                        outputTypes, outputValues);
    assert(inputTypes.size() == inputValues.size());
    assert(outputTypes.size() == outputValues.size());

    if (!outputTypes.empty()) {
      func::FuncOp foldFunc =
          buildFoldFunc(context, builder, topOp, "runtime_fold", constOps,
                        inputTypes, inputValues, outputTypes, outputValues);
      (void)foldFunc;
      canonicalizeAndClean(context, foldFunc.getOperation());

      // ===== build computing function =====
      modifyComputeFunc(context, builder, topOp, topFunc, block,
                        constArgsIndexes, outputTypes, outputValues);
    }
  }

  canonicalizeAndClean(context, topOp);
  topOp->dump();
}

std::unique_ptr<Pass> createConstantTensorFoldingPass() {
  return std::make_unique<ConstantTensorFolding>();
}

} // namespace gc
} // namespace mlir
