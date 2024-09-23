//===-- AddContextArg.cpp - Add context argument ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::gc {
#define GEN_PASS_DECL_ADDCONTEXTARG
#define GEN_PASS_DEF_ADDCONTEXTARG
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

using namespace mlir;

namespace {
struct AddContextArg final : gc::impl::AddContextArgBase<AddContextArg> {
  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal()) {
      return;
    }

    auto funcType = func.getFunctionType();
    auto argTypes = llvm::to_vector<8>(funcType.getInputs());
    auto resultTypes = llvm::to_vector<1>(funcType.getResults());
    auto ctx = func->getContext();
    auto newArgType = MemRefType::get({}, IntegerType::get(ctx, 8));
    argTypes.emplace_back(newArgType);
    auto newFuncType = FunctionType::get(ctx, argTypes, resultTypes);
    func.setType(newFuncType);
    func.getBody().front().addArgument(newArgType, func.getLoc());

    // Find all function calls and append the last argument of the current
    // function to the call.
    auto module = func->getParentOfType<ModuleOp>();
    func.walk([&](func::CallOp call) {
      // If the function to be called is defined in the current module, then the
      // context arg will be added to this function signature either and, thus,
      // wee need add the context arg to the function call.
      if (auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
          !callee || callee.isExternal()) {
        return;
      }
      auto args = llvm::to_vector<8>(call.getOperands());
      args.emplace_back(func.getArgument(func.getNumArguments() - 1));
      call->setOperands(args);
    });
  }
};
} // namespace
