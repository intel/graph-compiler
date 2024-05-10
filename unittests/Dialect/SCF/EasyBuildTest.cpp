//===- EasyBuildTest.cpp - Tests SCF Op Easy builders ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Arith/Utils/EasyBuild.h"
#include "gc/IR/EasyBuildSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::easybuild;

namespace {
class SCFTest : public ::testing::Test {
protected:
  SCFTest() {
    context.getOrLoadDialect<scf::SCFDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
  }

  mlir::MLIRContext context;
};
} // namespace

TEST_F(SCFTest, EasyBuild) {
  OpBuilder builder{&context};
  auto loc = builder.getUnknownLoc();
  EasyBuilder b{builder, loc, true};
  auto func = builder.create<func::FuncOp>(
      loc, "funcname",
      FunctionType::get(&context, {builder.getIndexType()},
                        {builder.getIndexType()}));

  builder.setInsertionPointToStart(func.addEntryBlock());
  auto init_val = b.wrap<EBUnsigned>(func.getArgument(0)) + b.toIndex(10);
  auto loop = builder.create<scf::ForOp>(loc, /*lo*/ b.toIndex(0),
                                         /*upper*/ b.toIndex(10),
                                         /*step*/ b.toIndex(1),
                                         /*ind_var*/ ValueRange{init_val});
  EB_for(auto &&[idx, redu] : forRangeIn<EBUnsigned, EBUnsigned>(b, loop)) {
    auto idx2 = idx + b.toIndex(1);
    b.F<scf::YieldOp, void>(ValueRange{idx2 + redu});
  }
  builder.create<func::ReturnOp>(loc, loop.getResult(0));
  std::string out;
  llvm::raw_string_ostream os{out};
  os << func;

  const char *expected =
      R"mlir(func.func @funcname(%arg0: index) -> index {
  %c10 = arith.constant 10 : index
  %0 = arith.addi %arg0, %c10 : index
  %c1 = arith.constant 1 : index
  %c10_0 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %1 = scf.for %arg1 = %c0 to %c10_0 step %c1 iter_args(%arg2 = %0) -> (index) {
    %c1_1 = arith.constant 1 : index
    %2 = arith.addi %arg1, %c1_1 : index
    %3 = arith.addi %2, %arg2 : index
    scf.yield %3 : index
  }
  return %1 : index
})mlir";
  ASSERT_EQ(out, expected);
}
