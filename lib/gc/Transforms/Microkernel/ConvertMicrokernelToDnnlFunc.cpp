//===-- ConvertMicrokernelToDnnlFunc.cpp - Lower to dnnl funcs --*- C++ -*-===//
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

#include "gc/Transforms/Microkernel/BrgemmRuntimeUtils.h"
#include "gc/Transforms/Microkernel/MicrokernelPasses.h"
#include "gc/Transforms/Utils/ValueUtils.h"

namespace mlir::microkernel {
#define GEN_PASS_DEF_CONVERTMICROKERNELTODNNLFUNC
#include "gc/Transforms/Microkernel/MicrokernelPasses.h.inc"

#define DEBUG_TYPE "convert-microkernel-to-dnnl-func"

static func::CallOp createFuncCall(RewriterBase &rewriter, Location loc,
                                   ModuleOp module, const std::string &funcName,
                                   ArrayRef<Value> operands,
                                   ArrayRef<Type> operandTypes,
                                   ArrayRef<Type> resultTypes) {
  FlatSymbolRefAttr fnName = SymbolRefAttr::get(module->getContext(), funcName);
  auto fnType = rewriter.getFunctionType(operandTypes, resultTypes);

  if (!module.lookupSymbol(fnName.getAttr())) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), fnType);
    funcOp.setPrivate();
  }

  func::CallOp call = rewriter.create<func::CallOp>(loc, fnName.getValue(),
                                                    resultTypes, operands);
  return call;
}

class ConvertBrgemmDispatchOpRewriter
    : public OpRewritePattern<microkernel::BrgemmDispatchOp> {
public:
  using OpRewritePattern<microkernel::BrgemmDispatchOp>::OpRewritePattern;
  // runtime func for dnnl brgemm dispatch:
  // int64_t dnnl_brgemm_dispatch(int64_t M, int64_t N, int64_t K, int64_t LDA,
  // int64_t LDB, int64_t LDC, int64_t stride_a, int64_t stride_b, float beta,
  // int64_t dtypeA, int64_t dtypeB);
  LogicalResult matchAndRewrite(microkernel::BrgemmDispatchOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();

    SmallVector<Value, 10> operands;
    SmallVector<Type, 10> operandTypes;
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    FloatType float32 = FloatType::getF32(rewriter.getContext());

    // M, N, K, LDA, LDB, LDC, stride_a, stride_b
    // they are in the same order with BrgemmDispatchOp inputs
    ArrayRef<int64_t> inputs = op.getInputsAttr().asArrayRef();
    for (auto input : inputs) {
      auto attr = IntegerAttr::get(rewriter.getI64Type(), input);
      operands.push_back(
          rewriter.create<arith::ConstantOp>(loc, integer64, attr));
      operandTypes.push_back(integer64);
    }

    // beta
    auto flags = op.getFlagsAttr();
    float beta = 1.0f;
    for (auto flag : flags) {
      auto brgemmFlag = dyn_cast_or_null<microkernel::BrgemmFlagsAttr>(flag);
      if (!brgemmFlag)
        return rewriter.notifyMatchFailure(op, "unknown flag for BRGEMM");
      if (brgemmFlag.getValue() == BrgemmFlags::LIST)
        return rewriter.notifyMatchFailure(
            op, "addr mode BRGEMM not supported yet");
      if (brgemmFlag.getValue() == BrgemmFlags::BETA_0)
        beta = 0.0f;
    }
    auto betaAttr = FloatAttr::get(rewriter.getF32Type(), beta);
    operands.push_back(
        rewriter.create<arith::ConstantOp>(loc, float32, betaAttr));
    operandTypes.push_back(float32);

    // dtypeA, dtypeB
    auto dtypes = op.getDataType();
    if (dtypes.size() != 2)
      return rewriter.notifyMatchFailure(
          op, "invalid number of DataType for BRGEMM");
    auto dtypeAAttr = IntegerAttr::get(rewriter.getI64Type(),
                                       getDnnlDataTypeVal(rewriter, dtypes[0]));
    auto dtypeBAttr = IntegerAttr::get(rewriter.getI64Type(),
                                       getDnnlDataTypeVal(rewriter, dtypes[1]));
    operands.push_back(
        rewriter.create<arith::ConstantOp>(loc, integer64, dtypeAAttr));
    operandTypes.push_back(integer64);
    operands.push_back(
        rewriter.create<arith::ConstantOp>(loc, integer64, dtypeBAttr));
    operandTypes.push_back(integer64);

    func::CallOp call =
        createFuncCall(rewriter, loc, module, DNNL_BRGEMM_DISPATCH_NAME,
                       operands, operandTypes, {integer64});
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

class ConvertBrgemmPrologueOpRewriter
    : public OpRewritePattern<microkernel::BrgemmPrologueOp> {
public:
  using OpRewritePattern<microkernel::BrgemmPrologueOp>::OpRewritePattern;
  // dnnl runtime func for brgemm set hw context:
  // void dnnl_brgemm_tileconfig(int64_t kernel_idx);
  LogicalResult matchAndRewrite(microkernel::BrgemmPrologueOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    func::CallOp call =
        createFuncCall(rewriter, loc, module, DNNL_BRGEMM_TILECFG_NAME,
                       op.getInputs(), {integer64}, {});
    rewriter.replaceOp(op, call);
    return success();
  }
};

class ConvertBrgemmOpRewriter : public OpRewritePattern<microkernel::BrgemmOp> {
public:
  using OpRewritePattern<microkernel::BrgemmOp>::OpRewritePattern;
  // runtime func for stride mode dnnl brgemm execution:
  // void dnnl_brgemm_execute(int64_t kernel, void *A, uint64_t A_offset, void
  // *B, uint64_t B_offset, void *C, uint64_t C_offset, int num)
  LogicalResult matchAndRewrite(microkernel::BrgemmOp op,
                                PatternRewriter &rewriter) const final {
    // currently only support stride mode, directly call it
    // TODO(haixin): support addr mode execution, through detecting dispatch
    // target

    auto context = rewriter.getContext();
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();

    SmallVector<Value, 10> operands;
    SmallVector<Type, 10> operandTypes;

    auto raw_operands = op->getOperands();
    size_t raw_op_cnt = 0;
    for (Value operand : raw_operands) {
      if (raw_op_cnt++ >= 5) {
        // drop the last operand for `addr list length`
        break;
      }
      Type operandType = operand.getType();
      if (auto memrefType = dyn_cast<MemRefType>(operandType)) {
        Type basePtrType = LLVM::LLVMPointerType::get(context);
        auto [ptr, offset] = utils::getPtrAndOffset(rewriter, operand);
        operands.push_back(ptr);
        operands.push_back(offset);
        operandTypes.push_back(basePtrType);
        operandTypes.push_back(rewriter.getIndexType()); // offset
      } else {
        operands.push_back(operand);
        operandTypes.push_back(operand.getType());
      }
    }

    createFuncCall(rewriter, loc, module, DNNL_BRGEMM_EXECUTE_NAME, operands,
                   operandTypes, {});
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertBrgemmEpilogueOpRewriter
    : public OpRewritePattern<microkernel::BrgemmEpilogueOp> {
public:
  using OpRewritePattern<microkernel::BrgemmEpilogueOp>::OpRewritePattern;
  // dnnl runtime func for brgemm release hw context:
  // void dnnl_brgemm_tilerelease();
  LogicalResult matchAndRewrite(microkernel::BrgemmEpilogueOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    func::CallOp call = createFuncCall(
        rewriter, loc, module, DNNL_BRGEMM_TILERELEASE_NAME, {}, {}, {});
    rewriter.replaceOp(op, call);
    return success();
  }
};

class ConvertMicrokernelToDnnlFunc
    : public impl::ConvertMicrokernelToDnnlFuncBase<
          ConvertMicrokernelToDnnlFunc> {
public:
  using impl::ConvertMicrokernelToDnnlFuncBase<
      ConvertMicrokernelToDnnlFunc>::ConvertMicrokernelToDnnlFuncBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns
        .add<ConvertBrgemmDispatchOpRewriter, ConvertBrgemmPrologueOpRewriter,
             ConvertBrgemmOpRewriter, ConvertBrgemmEpilogueOpRewriter>(
            &getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace mlir::microkernel
