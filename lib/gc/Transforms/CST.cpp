//===- CST.cpp - Constant Subgraph Transform -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs a constant subgraph transform in MLIR.
//
//===----------------------------------------------------------------------===//

#include <deque>
#include <unordered_set>

#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
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

#include "gc/ExecutionEngine/CPURuntime/ConstantCache.hpp"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CST
#include "gc/Transforms/Passes.h.inc"
} // namespace gc

using namespace mlir;

namespace gc {

struct CST : public impl::CSTBase<CST> {
  void runOnOperation() override;
};

bool isInConstantSubgraph(Operation *op) {
  auto opNamespace = op->getDialect()->getNamespace();
  if (opNamespace == linalg::LinalgDialect::getDialectNamespace() ||
      opNamespace == tensor::TensorDialect::getDialectNamespace() ||
      opNamespace == arith::ArithDialect::getDialectNamespace()) {
    if (op->getAttr("onednn_graph.in_const_subgraph")) {
      return true;
    }
  }
  return false;
}

int64_t getTensorSize(TensorType t) {
  Type eleType = t.getElementType();
  unsigned bitWidth = eleType.getIntOrFloatBitWidth() / 8; // bytes
  ArrayRef<int64_t> shape = t.getShape();
  int64_t size = bitWidth;
  for (auto s : shape) {
    size *= s;
  }
  return size;
}

bool singleOperand(Operation *op) {
  if (op->getNumOperands() > 1) {
    Value firstOperand = op->getOperand(0);
    for (int64_t i = 1; i < op->getNumOperands(); ++i) {
      Value operand = op->getOperand(i);
      if (firstOperand == operand) {
        continue;
      }
      auto parentOp = operand.getDefiningOp();
      if (parentOp && !isa<tensor::EmptyOp>(parentOp)) {
        return false;
      }
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
    if (!affineMap.isIdentity()) {
      return false;
    }
  }

  SmallVector<utils::IteratorType> iterTypes = linalgOp.getIteratorTypesArray();
  for (auto &iterType : iterTypes) {
    if (iterType != utils::IteratorType::parallel) {
      return false;
    }
  }

  if (op->getNumOperands() > 1) {
    // int64_t numInputs = linalgOp.getNumDpsInputs();
    int64_t numInits = linalgOp.getNumDpsInits();
    // definingOp of init should be tensor.empty()
    for (int64_t i = 0; i < numInits; ++i) {
      OpOperand *outOperand = linalgOp.getDpsInitOperand(i);
      auto parentOp = outOperand->get().getDefiningOp();
      if (!isa<tensor::EmptyOp>(parentOp)) {
        return false;
      }
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
      if (isInConstantSubgraph(bcOp)) {
        constBcOps.push_back(bcOp);
      }
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
      if (currOp->getNumOperands() != 1) {
        break;
      }
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
      if (currOp->getNumResults() != 1 || !currOp->hasOneUse()) {
        break;
      }
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
    if (postOps.empty()) {
      continue;
    }

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

    for (auto it = postOps.rbegin(); it != postOps.rend(); ++it) {
      (*it)->erase();
    }
  }
}

static constexpr int DATA_SIZE_EXPANDING_THRESHOLD = 8;

// get from dnnl_graph_compiler_context
// void *allocator(size_t size) { return std::aligned_alloc(64, size); }
// void deallocator(void *ptr) { std::free(ptr); }

std::shared_ptr<ConstCacheProxy> createConstCacheProxy(size_t size) {
  // simply allocate buffer and return
  std::shared_ptr<void> base = std::shared_ptr<void>{
      std::aligned_alloc(64, size), [](void *p) { std::free(p); }};
  return std::make_shared<ConstCacheProxy>(base, base.get(), size, true);
}

size_t divideAndCeil(size_t x, size_t y) { return (x + y - 1) / y; }

// Manager
struct constGraphTensorCacheManager {
  // dnnl_graph_compiler_context *ctx;

  uint64_t cachedTensorGlobalId = 0;

  // singleton
  static std::shared_ptr<constGraphTensorCacheManager> get() {
    static std::shared_ptr<constGraphTensorCacheManager> c =
        std::make_shared<constGraphTensorCacheManager>();
    return c;
  }

  // alloc and set the buf_base_ and offset_ attributes of cache
  std::vector<uint64_t> alloc(std::vector<size_t> buffersSize) {
    size_t totalSize = 0;
    for (size_t i = 0; i < buffersSize.size(); i++) {
      totalSize += divideAndCeil(buffersSize[i], 64) * 64;
    }
    llvm::dbgs() << "Alloc total size: " << totalSize << '\n';
    auto base = createConstCacheProxy(totalSize);
    std::vector<uint64_t> globalIds(buffersSize.size());
    size_t offset = 0;
    for (size_t i = 0; i < buffersSize.size(); i++) {
      llvm::dbgs() << "Alloc offset: " << offset << '\n';
      regCachedTensor(cachedTensorGlobalId, base, offset);
      globalIds[i] = cachedTensorGlobalId;
      ++cachedTensorGlobalId;
      offset += divideAndCeil(buffersSize[i], 64) * 64;
    }
    return globalIds;
  }
};

static void addGlobalI32(ModuleOp module, Location loc, OpBuilder &builder,
                         StringRef name, int32_t value) {
  OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto type = IntegerType::get(builder.getContext(), 32);
  LLVM::GlobalOp global = builder.create<LLVM::GlobalOp>(
      loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name,
      builder.getI32IntegerAttr(value),
      /*alignment=*/0);
}

static void addGlobalI64Array(ModuleOp module, Location loc, OpBuilder &builder,
                              StringRef name, ArrayRef<int64_t> array) {
  OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto type = LLVM::LLVMArrayType::get(
      IntegerType::get(builder.getContext(), 64), array.size());
  LLVM::GlobalOp global = builder.create<LLVM::GlobalOp>(
      loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name,
      builder.getI64ArrayAttr(array),
      /*alignment=*/0);
}

static void addGlobalI32Array(ModuleOp module, Location loc, OpBuilder &builder,
                              StringRef name, ArrayRef<int32_t> array) {
  OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto type = LLVM::LLVMArrayType::get(
      IntegerType::get(builder.getContext(), 32), array.size());
  LLVM::GlobalOp global = builder.create<LLVM::GlobalOp>(
      loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name,
      builder.getI32ArrayAttr(array),
      /*alignment=*/0);
}

// Operate on tensors. Create fold() and compute() on module. The
// folded weights and first-run flag is maintained by upper-level runtime.
void CST::runOnOperation() {
  Operation *topOp = getOperation();
  MLIRContext *context = topOp->getContext();
  // A ModuleOp contains a single region, which contains a single block.
  auto moduleOp = dyn_cast<ModuleOp>(topOp);
  SymbolTable symbolTable(moduleOp);
  auto &topFunc =
      topOp->getRegions().front().getBlocks().front().getOperations().front();
  OpBuilder builder(context);

  auto topFuncAttr = topFunc.getAttrDictionary();
  std::optional<NamedAttribute> constArgs =
      topFuncAttr.getNamed("onednn_graph.const_args");
  std::unordered_set<int> constArgsIndexes;
  if (constArgs.has_value()) {
    ArrayAttr constArgsArray = llvm::dyn_cast<ArrayAttr>(constArgs->getValue());
    for (auto id : constArgsArray) {
      constArgsIndexes.insert(llvm::cast<IntegerAttr>(id).getInt());
    }
  } else {
    return;
  }
  if (constArgsIndexes.empty()) {
    return;
  }

  Region &region = topFunc.getRegions().front();
  Block &block = region.getBlocks().front();

  postponeBroadcast(block);

  SmallVector<Operation *> constOps;
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isInConstantSubgraph(&op)) {
      constOps.push_back(&op);
    }
  }

  std::string funcName("fold");
  SmallVector<Type> inputTypes; // types of constant weights
  // values of constant weights in original block
  SmallVector<Value> inputValues;
  SmallVector<Type> outputTypes; // types of folded constant weights
  // values of folded constant weights in original block
  SmallVector<Value> outputValues;
  Value v;
  // TODO: solve complicated topology. Currently we only handle simple topology
  // where one constant weight input will and only will produce one constant
  // output and each constant weight only contributes to one constant output.
  for (size_t id = 0; id < block.getNumArguments(); ++id) {
    if (constArgsIndexes.count(id) == 1) {
      auto arg = block.getArgument(id);
      if (!isa<TensorType>(arg.getType())) {
        continue;
      }
      inputTypes.push_back(arg.getType());
      v = dyn_cast<Value>(arg);
      inputValues.push_back(v);
      SmallVector<Value> valuesOnTheWay = {v}; // the constant tensors
      // For v -> pack1 -> pack2 -> matmul, we need the type of output of pack2
      while (!v.getUsers().empty()) {
        // v.getUsers().size() should be 1
        Operation *user = *(v.getUsers().begin());
        // If user is not const or user has multiple operand, we reach the end
        if (!isInConstantSubgraph(user) || !singleOperand(user)) {
          outputTypes.push_back(v.getType());
          outputValues.push_back(v);
          break;
        }
        // user should has only 1 output value
        OpResult result = *(user->result_begin());
        v = dyn_cast<Value>(result);
        valuesOnTheWay.push_back(v);
      }

      // If data size of outputValue is too greater than size of inputValue, do
      // not fold it. Compare data size changes during traverse to find the last
      // op that satisfies this condition.
      int64_t initSize =
          getTensorSize(dyn_cast<TensorType>(valuesOnTheWay[0].getType()));
      if (!isa<TensorType>(outputTypes.back()) ||
          initSize * DATA_SIZE_EXPANDING_THRESHOLD <
              getTensorSize(dyn_cast<TensorType>(outputTypes.back()))) {
        size_t lastIdx = 0;
        for (size_t i = 1; i < valuesOnTheWay.size(); ++i) {
          int64_t size =
              getTensorSize(dyn_cast<TensorType>(valuesOnTheWay[i].getType()));
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
  if (inputTypes.size() != outputTypes.size()) {
    return;
  }

  FunctionType foldFuncType =
      FunctionType::get(context, inputTypes, outputTypes);
  auto foldFunc =
      builder.create<func::FuncOp>(topFunc.getLoc(), funcName, foldFuncType);
  Block *foldBlock = foldFunc.addEntryBlock();
  // values of folded constant weights in foldBlock
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

  // Allocate buffer for outputValuesInFold
  std::vector<size_t> buffersSize;
  for (Value &tensor : outputValuesInFold) {
    llvm::dbgs() << "Allocate buffer for tensor: " << tensor << "\n";
    buffersSize.push_back(
        getTensorSize(dyn_cast<TensorType>(tensor.getType())));
  }
  auto manager = constGraphTensorCacheManager::get();
  SmallVector<int64_t> globalIndexes;
  for (auto id : manager->alloc(buffersSize)) {
    globalIndexes.push_back(id);
  }
  globalIndexes.insert(globalIndexes.begin(), globalIndexes.size());
  addGlobalI64Array(moduleOp, moduleOp.getLoc(), builder, "__fold_buffer_ids",
                    globalIndexes);

  foldFunc.setVisibility(SymbolTable::Visibility::Public);
  moduleOp.push_back(foldFunc);
  symbolTable.insert(foldFunc);

  SmallVector<int32_t> foldArgs;
  SmallVector<int32_t> foldIds;
  SmallVector<int32_t> computeArgs;

  // modify the BlockArguments of block
  size_t oriNumArgs = block.getNumArguments();
  size_t argIdx = 0;
  for (size_t id = 0; id < oriNumArgs; ++id) {
    if (constArgsIndexes.count(id) == 1) {
      foldArgs.push_back(id);
      foldIds.push_back(argIdx + oriNumArgs);
      computeArgs.push_back(argIdx + oriNumArgs);
      auto loc = block.getArgument(id).getLoc();
      BlockArgument foldArg =
          block.insertArgument(id, outputTypes[argIdx], loc);
      outputValues[argIdx].replaceUsesWithIf(foldArg, [&](OpOperand &val) {
        Operation *op = val.getOwner();
        return op->getBlock() == &block;
      });

      std::deque<Value> dq;
      SmallVector<Operation *> opsToErase;
      std::unordered_set<Operation *> opsToEraseSet;
      dq.push_back(block.getArgument(id + 1));
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
      block.eraseArgument(id + 1);
      ++argIdx;
    } else {
      computeArgs.push_back(id);
    }
  }

  for (auto id : foldIds) {
    foldArgs.insert(foldArgs.end(), id);
  }
  foldArgs.insert(foldArgs.begin(), foldArgs.size());
  addGlobalI32Array(moduleOp, moduleOp.getLoc(), builder, "__fold_args",
                    foldArgs);

  computeArgs.insert(computeArgs.begin(), computeArgs.size());
  addGlobalI32Array(moduleOp, moduleOp.getLoc(), builder, "__compute_args",
                    computeArgs);

  addGlobalI32(moduleOp, moduleOp.getLoc(), builder, "__num_orig_num_args",
               oriNumArgs);

  // modify the compute func signature
  func::FuncOp computeFunc = cast<func::FuncOp>(topFunc);
  FunctionType computeFuncType = computeFunc.getFunctionType();
  computeFunc.setType(FunctionType::get(context, block.getArgumentTypes(),
                                        computeFuncType.getResults()));

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
  for (auto &op : block.getOperations()) {
    if (op.getAttr("onednn_graph.in_const_subgraph")) {
      op.removeAttr("onednn_graph.in_const_subgraph");
    }
  }
  for (auto &op : foldBlock->getOperations()) {
    if (op.getAttr("onednn_graph.in_const_subgraph")) {
      op.removeAttr("onednn_graph.in_const_subgraph");
    }
  }
}

std::unique_ptr<Pass> createCSTPass() { return std::make_unique<CST>(); }

} // namespace gc
} // namespace mlir
