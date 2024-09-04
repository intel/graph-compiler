//===-- EarlyDispatchMicrokernel.cpp - Dispatch before runtime --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <sstream>

#include "gc/Transforms/Microkernel/BrgemmRuntimeUtils.h"
#include "gc/Transforms/Microkernel/MicrokernelPasses.h"
#include "gc/Transforms/Utils/ValueUtils.h"
#include "oneapi/dnnl/dnnl_types.h"

namespace mlir::microkernel {
#define GEN_PASS_DEF_EARLYDISPATCHMICROKERNEL
#include "gc/Transforms/Microkernel/MicrokernelPasses.h.inc"

#define DEBUG_TYPE "early-dispatch-microkernel"

static FailureOr<std::string>
createGlobalKernelHandleName(RewriterBase &rewriter,
                             microkernel::BrgemmDispatchOp op) {
  // TODO(haixin): Add runtime backend type to global name
  std::stringstream ss;
  ss << "g_dispatched_microkernel_brgemm";

  bool isInit = false;
  bool isStrideMode = false;
  auto flags = op.getFlagsAttr();
  for (auto flag : flags) {
    auto brgemmFlag = dyn_cast_or_null<microkernel::BrgemmFlagsAttr>(flag);
    if (!brgemmFlag)
      return failure();
    if (brgemmFlag.getValue() == BrgemmFlags::LIST)
      // TODO(haixin): Currently not supported. Support list brgemm in the
      // future
      return failure();
    else if (brgemmFlag.getValue() == BrgemmFlags::STRIDE)
      isStrideMode = true;
    else if (brgemmFlag.getValue() == BrgemmFlags::BETA_0)
      isInit = true;
  }
  if (isStrideMode)
    ss << "_stride";
  else
    ss << "_list";
  if (isInit)
    ss << "_init";

  // M, N, K, LDA, LDB, LDC, stride_a, stride_b
  // they are in the same order with BrgemmDispatchOp inputs
  // TODO(haixin): Add order enforcement machanism for BrgemmDispatchOp
  ArrayRef<int64_t> inputs = op.getInputsAttr().asArrayRef();
  for (auto input : inputs) {
    ss << "_" << input;
  }

  // dtypeA, dtypeB
  auto dtypes = op.getDataType();
  if (dtypes.size() != 2)
    return failure();
  ss << "_" << getDnnlDataTypeVal(rewriter, dtypes[0]);
  ss << "_" << getDnnlDataTypeVal(rewriter, dtypes[1]);

  return ss.str();
}

// get or create global kernel handle with initializer, identified by
// `kernelName`
static FailureOr<LLVM::GlobalOp>
getOrCreateGlobalKernelHandle(RewriterBase &rewriter, ModuleOp module,
                              const std::string &kernelName,
                              microkernel::BrgemmDispatchOp op) {
  // Create the global at the entry of the module
  LLVM::GlobalOp global = module.lookupSymbol<LLVM::GlobalOp>(kernelName);
  if (global)
    return global;

  auto global_type = op.getResults().getType();
  FlatSymbolRefAttr ctorName =
      SymbolRefAttr::get(module->getContext(), kernelName + "_ctor");
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(ctorName.getAttr()))
    return failure();

  OpBuilder::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  global = rewriter.create<LLVM::GlobalOp>(
      module.getLoc(), global_type, /*isConstant=*/false,
      LLVM::Linkage::Internal, kernelName, Attribute(),
      /*alignment=*/0);

  // create ctor for this global, which needs to be LLVMFuncOp
  LLVM::LLVMFuncOp ctorFunc = rewriter.create<LLVM::LLVMFuncOp>(
      module.getLoc(), ctorName.getValue(),
      LLVM::LLVMFunctionType::get(global_type, {}, false));

  Location loc = ctorFunc.getLoc();
  Block *entryBlock = ctorFunc.addEntryBlock(rewriter);
  rewriter.setInsertionPointToEnd(entryBlock);

  auto dispatch = op.clone();
  rewriter.insert(dispatch);
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
  rewriter.create<LLVM::StoreOp>(loc, dispatch.getResults(), globalPtr);
  rewriter.create<LLVM::ReturnOp>(loc, dispatch.getResults());

  // initialize the gloabl with global_ctors, as the initializer of global
  // does not allow side effect
  rewriter.setInsertionPointToStart(module.getBody());
  LLVM::GlobalCtorsOp global_ctors = nullptr;
  for (auto &op : module->getRegion(0).front()) {
    auto ctorOp = dyn_cast<LLVM::GlobalCtorsOp>(op);
    if (ctorOp) {
      global_ctors = ctorOp;
      break;
    }
  }

  SmallVector<Attribute> ctorRefs;
  SmallVector<Attribute> priorities;
  if (global_ctors) {
    auto ctorRefsAttr = global_ctors.getCtors();
    auto prioritiesAttr = global_ctors.getPriorities();
    for (auto &&[ctor, prior] : llvm::zip(ctorRefsAttr, prioritiesAttr)) {
      ctorRefs.push_back(ctor);
      priorities.push_back(prior);
    }
    LLVM_DEBUG(llvm::dbgs()
               << "After append ctors: " << ctorRefs.size() << "\n");
  }
  ctorRefs.push_back(ctorName);
  // Set new ctor's priority to lowest
  priorities.push_back(IntegerAttr::get(rewriter.getI32Type(), INT_MAX));
  if (global_ctors) {
    LLVM_DEBUG(llvm::dbgs() << "Replace existing ctors\n");
    // If there's existing ctors
    rewriter.replaceOpWithNewOp<LLVM::GlobalCtorsOp>(
        global_ctors, rewriter.getArrayAttr(ctorRefs),
        rewriter.getArrayAttr(priorities));
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Create new ctor\n");
    rewriter.create<LLVM::GlobalCtorsOp>(module.getLoc(),
                                         rewriter.getArrayAttr(ctorRefs),
                                         rewriter.getArrayAttr(priorities));
  }
  return global;
}

class EarlyDispatchBrgemmRewriter
    : public OpRewritePattern<microkernel::BrgemmDispatchOp> {
public:
  using OpRewritePattern<microkernel::BrgemmDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(microkernel::BrgemmDispatchOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    func::FuncOp func = op->template getParentOfType<func::FuncOp>();

    auto globalKernelName = createGlobalKernelHandleName(rewriter, op);
    if (failed(globalKernelName)) {
      return rewriter.notifyMatchFailure(
          op, "Failed to create global kernel handle name");
    }

    // Generate kernel handle global name
    auto globalKernel =
        getOrCreateGlobalKernelHandle(rewriter, module, *globalKernelName, op);
    if (failed(globalKernel)) {
      return rewriter.notifyMatchFailure(
          op, "Failed to create global kernel handle");
    }

    // Inject global val loading into start of function
    auto funcBlock = &func.getBody().front();
    rewriter.setInsertionPointToStart(funcBlock);
    Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, *globalKernel);
    Value globalVal = rewriter.create<LLVM::LoadOp>(
        loc, op.getResults().getType(), globalPtr);
    rewriter.moveOpAfter(op, funcBlock, funcBlock->begin());
    rewriter.replaceOp(op, globalVal);
    return success();
  }
};

class EarlyDispatchMicrokernel
    : public impl::EarlyDispatchMicrokernelBase<EarlyDispatchMicrokernel> {
public:
  using impl::EarlyDispatchMicrokernelBase<
      EarlyDispatchMicrokernel>::EarlyDispatchMicrokernelBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<EarlyDispatchBrgemmRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));

    // Ignore newly created Ops
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), patternSet, config)))
      signalPassFailure();
  }
};

} // namespace mlir::microkernel
