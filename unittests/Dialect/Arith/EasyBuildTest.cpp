//===- EasyBuildTest.cpp - Tests Arith Op Easy builders -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Arith/Utils/EasyBuild.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::easybuild;

namespace {
class ArithTest : public ::testing::Test {
protected:
  ArithTest() {
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
  }

  mlir::MLIRContext context;
};
} // namespace

TEST_F(ArithTest, EasyBuildConst) {
  OpBuilder builder{&context};
  auto loc = builder.getUnknownLoc();
  EasyBuilder b{builder, loc};
  auto func = builder.create<func::FuncOp>(loc, "funcname",
                                           FunctionType::get(&context, {}, {}));

  builder.setInsertionPointToStart(func.addEntryBlock());
  auto i1 = b(true);
  i1 = b(false);
  auto i8 = b(int8_t(3));
  i8 = b(int8_t(-3));
  auto u8 = b(uint8_t(33));

  auto i16 = b(int16_t(33));
  i16 = b(int16_t(-33));
  auto u16 = b(uint16_t(33));

  auto i32 = b(int32_t(33));
  i32 = b(int32_t(-33));
  auto u32 = b(uint32_t(33));

  auto i64 = b(int64_t(33));
  i64 = b(int64_t(-33));
  auto u64 = b(uint64_t(33));

  auto idx = b.toIndex(23);

  {
    EasyBuilder b2{builder, loc, /*u64AsIndex*/ true};
    auto idx2 = b(uint64_t(33));
  }
  builder.create<func::ReturnOp>(loc);
  std::string out;
  llvm::raw_string_ostream os{out};
  os << func;

  const char *expected =
      R"mlir(func.func @funcname() {
  %true = arith.constant true
  %false = arith.constant false
  %c3_i8 = arith.constant 3 : i8
  %c-3_i8 = arith.constant -3 : i8
  %c33_i8 = arith.constant 33 : i8
  %c33_i16 = arith.constant 33 : i16
  %c-33_i16 = arith.constant -33 : i16
  %c33_i16_0 = arith.constant 33 : i16
  %c33_i32 = arith.constant 33 : i32
  %c-33_i32 = arith.constant -33 : i32
  %c33_i32_1 = arith.constant 33 : i32
  %c33_i64 = arith.constant 33 : i64
  %c-33_i64 = arith.constant -33 : i64
  %c33_i64_2 = arith.constant 33 : i64
  %c23 = arith.constant 23 : index
  %c33_i64_3 = arith.constant 33 : i64
  return
})mlir";
  ASSERT_EQ(out, expected);
}

#define SKIP_IF_UNEXPECTED_FP_SIZE()                                           \
  if constexpr (sizeof(float) != 4 || sizeof(double) != 8) {                   \
    GTEST_SKIP();                                                              \
  }

TEST_F(ArithTest, EasyBuildFloatConst) {
  SKIP_IF_UNEXPECTED_FP_SIZE()
  OpBuilder builder{&context};
  auto loc = builder.getUnknownLoc();
  EasyBuilder b{builder, loc};
  auto func = builder.create<func::FuncOp>(loc, "funcname",
                                           FunctionType::get(&context, {}, {}));

  builder.setInsertionPointToStart(func.addEntryBlock());
  auto a = b(1.0f);
  auto a2 = b(1.0);
  builder.create<func::ReturnOp>(loc);
  std::string out;
  llvm::raw_string_ostream os{out};
  os << func;
  const char *expected =
      R"mlir(func.func @funcname() {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f64
  return
})mlir";
  ASSERT_EQ(out, expected);
}

template <typename T1, typename T2>
static std::string composeIR(MLIRContext *context, T1 &&getA, T2 &&getB) {
  OpBuilder builder{context};
  auto loc = builder.getUnknownLoc();
  EasyBuilder b{builder, loc};
  auto func = builder.create<func::FuncOp>(
      loc, "funcname", FunctionType::get(builder.getContext(), {}, {}));
  builder.setInsertionPointToStart(func.addEntryBlock());
  auto A = getA(b);
  auto B = getB(b);
  auto v1 = A + B;
  v1 = A - B;
  v1 = A * B;
  v1 = A / B;
  v1 = A % B;
  v1 = A >> B;
  v1 = A << B;
  v1 = A & B;
  v1 = A | B;
  v1 = A ^ B;
  auto cmp = A < B;
  cmp = cmp & (A <= B);
  cmp = cmp & (A > B);
  cmp = cmp & (A >= B);
  cmp = cmp ^ (A == B);
  cmp = cmp ^ (A != B);
  builder.create<func::ReturnOp>(loc);

  std::string out;
  llvm::raw_string_ostream os{out};
  os << func;
  return out;
}

// check X+Y, where both X and Y are WrappedValues
TEST_F(ArithTest, EasyBuildSignedOperatorsBothValues) {
  auto out = composeIR(
      &context, [](EasyBuilder b) { return b(int32_t(33)); },
      [](EasyBuilder b) { return b(int32_t(31)); });
  const char *signedExpected =
      R"mlir(func.func @funcname() {
  %c33_i32 = arith.constant 33 : i32
  %c31_i32 = arith.constant 31 : i32
  %0 = arith.addi %c33_i32, %c31_i32 : i32
  %1 = arith.subi %c33_i32, %c31_i32 : i32
  %2 = arith.muli %c33_i32, %c31_i32 : i32
  %3 = arith.divsi %c33_i32, %c31_i32 : i32
  %4 = arith.remsi %c33_i32, %c31_i32 : i32
  %5 = arith.shrsi %c33_i32, %c31_i32 : i32
  %6 = arith.shli %c33_i32, %c31_i32 : i32
  %7 = arith.andi %c33_i32, %c31_i32 : i32
  %8 = arith.ori %c33_i32, %c31_i32 : i32
  %9 = arith.xori %c33_i32, %c31_i32 : i32
  %10 = arith.cmpi slt, %c33_i32, %c31_i32 : i32
  %11 = arith.cmpi sle, %c33_i32, %c31_i32 : i32
  %12 = arith.andi %10, %11 : i1
  %13 = arith.cmpi sgt, %c33_i32, %c31_i32 : i32
  %14 = arith.andi %12, %13 : i1
  %15 = arith.cmpi sge, %c33_i32, %c31_i32 : i32
  %16 = arith.andi %14, %15 : i1
  %17 = arith.cmpi eq, %c33_i32, %c31_i32 : i32
  %18 = arith.xori %16, %17 : i1
  %19 = arith.cmpi ne, %c33_i32, %c31_i32 : i32
  %20 = arith.xori %18, %19 : i1
  return
})mlir";
  ASSERT_EQ(out, signedExpected);
}

// check X+Y, where X is compile-time value (like 1) and Y is WrappedValue
TEST_F(ArithTest, EasyBuildSignedOperatorsLHSConst) {
  auto out = composeIR(
      &context, [](EasyBuilder b) { return int32_t(33); },
      [](EasyBuilder b) { return b(int32_t(31)); });
  const char *signedExpected =
      R"mlir(func.func @funcname() {
  %c31_i32 = arith.constant 31 : i32
  %c33_i32 = arith.constant 33 : i32
  %0 = arith.addi %c33_i32, %c31_i32 : i32
  %c33_i32_0 = arith.constant 33 : i32
  %1 = arith.subi %c33_i32_0, %c31_i32 : i32
  %c33_i32_1 = arith.constant 33 : i32
  %2 = arith.muli %c33_i32_1, %c31_i32 : i32
  %c33_i32_2 = arith.constant 33 : i32
  %3 = arith.divsi %c33_i32_2, %c31_i32 : i32
  %c33_i32_3 = arith.constant 33 : i32
  %4 = arith.remsi %c33_i32_3, %c31_i32 : i32
  %c33_i32_4 = arith.constant 33 : i32
  %5 = arith.shrsi %c33_i32_4, %c31_i32 : i32
  %c33_i32_5 = arith.constant 33 : i32
  %6 = arith.shli %c33_i32_5, %c31_i32 : i32
  %c33_i32_6 = arith.constant 33 : i32
  %7 = arith.andi %c33_i32_6, %c31_i32 : i32
  %c33_i32_7 = arith.constant 33 : i32
  %8 = arith.ori %c33_i32_7, %c31_i32 : i32
  %c33_i32_8 = arith.constant 33 : i32
  %9 = arith.xori %c33_i32_8, %c31_i32 : i32
  %c33_i32_9 = arith.constant 33 : i32
  %10 = arith.cmpi slt, %c33_i32_9, %c31_i32 : i32
  %c33_i32_10 = arith.constant 33 : i32
  %11 = arith.cmpi sle, %c33_i32_10, %c31_i32 : i32
  %12 = arith.andi %10, %11 : i1
  %c33_i32_11 = arith.constant 33 : i32
  %13 = arith.cmpi sgt, %c33_i32_11, %c31_i32 : i32
  %14 = arith.andi %12, %13 : i1
  %c33_i32_12 = arith.constant 33 : i32
  %15 = arith.cmpi sge, %c33_i32_12, %c31_i32 : i32
  %16 = arith.andi %14, %15 : i1
  %c33_i32_13 = arith.constant 33 : i32
  %17 = arith.cmpi eq, %c33_i32_13, %c31_i32 : i32
  %18 = arith.xori %16, %17 : i1
  %c33_i32_14 = arith.constant 33 : i32
  %19 = arith.cmpi ne, %c33_i32_14, %c31_i32 : i32
  %20 = arith.xori %18, %19 : i1
  return
})mlir";
  ASSERT_EQ(out, signedExpected);
}

// check X+Y, where Y is compile-time value (like 1) and X is WrappedValue
TEST_F(ArithTest, EasyBuildSignedOperatorsRHSConst) {
  auto out = composeIR(
      &context, [](EasyBuilder b) { return b(int32_t(33)); },
      [](EasyBuilder b) { return int32_t(31); });
  const char *signedExpected =
      R"mlir(func.func @funcname() {
  %c33_i32 = arith.constant 33 : i32
  %c31_i32 = arith.constant 31 : i32
  %0 = arith.addi %c33_i32, %c31_i32 : i32
  %c31_i32_0 = arith.constant 31 : i32
  %1 = arith.subi %c33_i32, %c31_i32_0 : i32
  %c31_i32_1 = arith.constant 31 : i32
  %2 = arith.muli %c33_i32, %c31_i32_1 : i32
  %c31_i32_2 = arith.constant 31 : i32
  %3 = arith.divsi %c33_i32, %c31_i32_2 : i32
  %c31_i32_3 = arith.constant 31 : i32
  %4 = arith.remsi %c33_i32, %c31_i32_3 : i32
  %c31_i32_4 = arith.constant 31 : i32
  %5 = arith.shrsi %c33_i32, %c31_i32_4 : i32
  %c31_i32_5 = arith.constant 31 : i32
  %6 = arith.shli %c33_i32, %c31_i32_5 : i32
  %c31_i32_6 = arith.constant 31 : i32
  %7 = arith.andi %c33_i32, %c31_i32_6 : i32
  %c31_i32_7 = arith.constant 31 : i32
  %8 = arith.ori %c33_i32, %c31_i32_7 : i32
  %c31_i32_8 = arith.constant 31 : i32
  %9 = arith.xori %c33_i32, %c31_i32_8 : i32
  %c31_i32_9 = arith.constant 31 : i32
  %10 = arith.cmpi slt, %c33_i32, %c31_i32_9 : i32
  %c31_i32_10 = arith.constant 31 : i32
  %11 = arith.cmpi sle, %c33_i32, %c31_i32_10 : i32
  %12 = arith.andi %10, %11 : i1
  %c31_i32_11 = arith.constant 31 : i32
  %13 = arith.cmpi sgt, %c33_i32, %c31_i32_11 : i32
  %14 = arith.andi %12, %13 : i1
  %c31_i32_12 = arith.constant 31 : i32
  %15 = arith.cmpi sge, %c33_i32, %c31_i32_12 : i32
  %16 = arith.andi %14, %15 : i1
  %c31_i32_13 = arith.constant 31 : i32
  %17 = arith.cmpi eq, %c33_i32, %c31_i32_13 : i32
  %18 = arith.xori %16, %17 : i1
  %c31_i32_14 = arith.constant 31 : i32
  %19 = arith.cmpi ne, %c33_i32, %c31_i32_14 : i32
  %20 = arith.xori %18, %19 : i1
  return
})mlir";
  ASSERT_EQ(out, signedExpected);
}

// check X+Y, where both X and Y are WrappedValues
TEST_F(ArithTest, EasyBuildUnsignedOperatorsBothValues) {
  auto out = composeIR(
      &context, [](EasyBuilder b) { return b(uint32_t(33)); },
      [](EasyBuilder b) { return b(uint32_t(31)); });
  const char *unsignedExpected =
      R"mlir(func.func @funcname() {
  %c33_i32 = arith.constant 33 : i32
  %c31_i32 = arith.constant 31 : i32
  %0 = arith.addi %c33_i32, %c31_i32 : i32
  %1 = arith.subi %c33_i32, %c31_i32 : i32
  %2 = arith.muli %c33_i32, %c31_i32 : i32
  %3 = arith.divui %c33_i32, %c31_i32 : i32
  %4 = arith.remui %c33_i32, %c31_i32 : i32
  %5 = arith.shrui %c33_i32, %c31_i32 : i32
  %6 = arith.shli %c33_i32, %c31_i32 : i32
  %7 = arith.andi %c33_i32, %c31_i32 : i32
  %8 = arith.ori %c33_i32, %c31_i32 : i32
  %9 = arith.xori %c33_i32, %c31_i32 : i32
  %10 = arith.cmpi ult, %c33_i32, %c31_i32 : i32
  %11 = arith.cmpi ule, %c33_i32, %c31_i32 : i32
  %12 = arith.andi %10, %11 : i1
  %13 = arith.cmpi ugt, %c33_i32, %c31_i32 : i32
  %14 = arith.andi %12, %13 : i1
  %15 = arith.cmpi uge, %c33_i32, %c31_i32 : i32
  %16 = arith.andi %14, %15 : i1
  %17 = arith.cmpi eq, %c33_i32, %c31_i32 : i32
  %18 = arith.xori %16, %17 : i1
  %19 = arith.cmpi ne, %c33_i32, %c31_i32 : i32
  %20 = arith.xori %18, %19 : i1
  return
})mlir";
  ASSERT_EQ(out, unsignedExpected);
}

// check X+Y, where X is compile-time value (like 1) and Y is WrappedValue
TEST_F(ArithTest, EasyBuildUnsignedOperatorsLHSConst) {
  auto out = composeIR(
      &context, [](EasyBuilder b) { return uint32_t(33); },
      [](EasyBuilder b) { return b(uint32_t(31)); });
  const char *unsignedExpected =
      R"mlir(func.func @funcname() {
  %c31_i32 = arith.constant 31 : i32
  %c33_i32 = arith.constant 33 : i32
  %0 = arith.addi %c33_i32, %c31_i32 : i32
  %c33_i32_0 = arith.constant 33 : i32
  %1 = arith.subi %c33_i32_0, %c31_i32 : i32
  %c33_i32_1 = arith.constant 33 : i32
  %2 = arith.muli %c33_i32_1, %c31_i32 : i32
  %c33_i32_2 = arith.constant 33 : i32
  %3 = arith.divui %c33_i32_2, %c31_i32 : i32
  %c33_i32_3 = arith.constant 33 : i32
  %4 = arith.remui %c33_i32_3, %c31_i32 : i32
  %c33_i32_4 = arith.constant 33 : i32
  %5 = arith.shrui %c33_i32_4, %c31_i32 : i32
  %c33_i32_5 = arith.constant 33 : i32
  %6 = arith.shli %c33_i32_5, %c31_i32 : i32
  %c33_i32_6 = arith.constant 33 : i32
  %7 = arith.andi %c33_i32_6, %c31_i32 : i32
  %c33_i32_7 = arith.constant 33 : i32
  %8 = arith.ori %c33_i32_7, %c31_i32 : i32
  %c33_i32_8 = arith.constant 33 : i32
  %9 = arith.xori %c33_i32_8, %c31_i32 : i32
  %c33_i32_9 = arith.constant 33 : i32
  %10 = arith.cmpi ult, %c33_i32_9, %c31_i32 : i32
  %c33_i32_10 = arith.constant 33 : i32
  %11 = arith.cmpi ule, %c33_i32_10, %c31_i32 : i32
  %12 = arith.andi %10, %11 : i1
  %c33_i32_11 = arith.constant 33 : i32
  %13 = arith.cmpi ugt, %c33_i32_11, %c31_i32 : i32
  %14 = arith.andi %12, %13 : i1
  %c33_i32_12 = arith.constant 33 : i32
  %15 = arith.cmpi uge, %c33_i32_12, %c31_i32 : i32
  %16 = arith.andi %14, %15 : i1
  %c33_i32_13 = arith.constant 33 : i32
  %17 = arith.cmpi eq, %c33_i32_13, %c31_i32 : i32
  %18 = arith.xori %16, %17 : i1
  %c33_i32_14 = arith.constant 33 : i32
  %19 = arith.cmpi ne, %c33_i32_14, %c31_i32 : i32
  %20 = arith.xori %18, %19 : i1
  return
})mlir";
  ASSERT_EQ(out, unsignedExpected);
}

// check X+Y, where Y is compile-time value (like 1) and X is WrappedValue
TEST_F(ArithTest, EasyBuildUnsignedOperatorsRHSConst) {
  auto out = composeIR(
      &context, [](EasyBuilder b) { return b(uint32_t(33)); },
      [](EasyBuilder b) { return uint32_t(31); });
  const char *unsignedExpected =
      R"mlir(func.func @funcname() {
  %c33_i32 = arith.constant 33 : i32
  %c31_i32 = arith.constant 31 : i32
  %0 = arith.addi %c33_i32, %c31_i32 : i32
  %c31_i32_0 = arith.constant 31 : i32
  %1 = arith.subi %c33_i32, %c31_i32_0 : i32
  %c31_i32_1 = arith.constant 31 : i32
  %2 = arith.muli %c33_i32, %c31_i32_1 : i32
  %c31_i32_2 = arith.constant 31 : i32
  %3 = arith.divui %c33_i32, %c31_i32_2 : i32
  %c31_i32_3 = arith.constant 31 : i32
  %4 = arith.remui %c33_i32, %c31_i32_3 : i32
  %c31_i32_4 = arith.constant 31 : i32
  %5 = arith.shrui %c33_i32, %c31_i32_4 : i32
  %c31_i32_5 = arith.constant 31 : i32
  %6 = arith.shli %c33_i32, %c31_i32_5 : i32
  %c31_i32_6 = arith.constant 31 : i32
  %7 = arith.andi %c33_i32, %c31_i32_6 : i32
  %c31_i32_7 = arith.constant 31 : i32
  %8 = arith.ori %c33_i32, %c31_i32_7 : i32
  %c31_i32_8 = arith.constant 31 : i32
  %9 = arith.xori %c33_i32, %c31_i32_8 : i32
  %c31_i32_9 = arith.constant 31 : i32
  %10 = arith.cmpi ult, %c33_i32, %c31_i32_9 : i32
  %c31_i32_10 = arith.constant 31 : i32
  %11 = arith.cmpi ule, %c33_i32, %c31_i32_10 : i32
  %12 = arith.andi %10, %11 : i1
  %c31_i32_11 = arith.constant 31 : i32
  %13 = arith.cmpi ugt, %c33_i32, %c31_i32_11 : i32
  %14 = arith.andi %12, %13 : i1
  %c31_i32_12 = arith.constant 31 : i32
  %15 = arith.cmpi uge, %c33_i32, %c31_i32_12 : i32
  %16 = arith.andi %14, %15 : i1
  %c31_i32_13 = arith.constant 31 : i32
  %17 = arith.cmpi eq, %c33_i32, %c31_i32_13 : i32
  %18 = arith.xori %16, %17 : i1
  %c31_i32_14 = arith.constant 31 : i32
  %19 = arith.cmpi ne, %c33_i32, %c31_i32_14 : i32
  %20 = arith.xori %18, %19 : i1
  return
})mlir";
  ASSERT_EQ(out, unsignedExpected);
}

template <typename T1, typename T2>
static std::string composeFPIR(MLIRContext *context, T1 &&getA, T2 &&getB) {
  OpBuilder builder{context};
  auto loc = builder.getUnknownLoc();
  EasyBuilder b{builder, loc};
  auto func = builder.create<func::FuncOp>(
      loc, "funcname", FunctionType::get(builder.getContext(), {}, {}));
  builder.setInsertionPointToStart(func.addEntryBlock());
  auto A = getA(b);
  auto B = getB(b);
  auto v1 = A + B;
  v1 = A - B;
  v1 = A * B;
  v1 = A / B;
  v1 = A % B;
  (void)-A;
  auto cmp = A < B;
  cmp = cmp & (A <= B);
  cmp = cmp & (A > B);
  cmp = cmp & (A >= B);
  cmp = cmp ^ (A == B);
  cmp = cmp ^ (A != B);
  builder.create<func::ReturnOp>(loc);

  std::string out;
  llvm::raw_string_ostream os{out};
  os << func;
  return out;
}

// check X+Y, where both X and Y are WrappedValues
TEST_F(ArithTest, EasyBuildFloatOperatorsValues) {
  SKIP_IF_UNEXPECTED_FP_SIZE()
  auto out = composeFPIR(
      &context, [](EasyBuilder b) { return b(33.0f); },
      [](EasyBuilder b) { return b(31.0f); });
  const char *expected =
      R"mlir(func.func @funcname() {
  %cst = arith.constant 3.300000e+01 : f32
  %cst_0 = arith.constant 3.100000e+01 : f32
  %0 = arith.addf %cst, %cst_0 : f32
  %1 = arith.subf %cst, %cst_0 : f32
  %2 = arith.mulf %cst, %cst_0 : f32
  %3 = arith.divf %cst, %cst_0 : f32
  %4 = arith.remf %cst, %cst_0 : f32
  %5 = arith.negf %cst : f32
  %6 = arith.cmpf olt, %cst, %cst_0 : f32
  %7 = arith.cmpf ole, %cst, %cst_0 : f32
  %8 = arith.andi %6, %7 : i1
  %9 = arith.cmpf ogt, %cst, %cst_0 : f32
  %10 = arith.andi %8, %9 : i1
  %11 = arith.cmpf oge, %cst, %cst_0 : f32
  %12 = arith.andi %10, %11 : i1
  %13 = arith.cmpf oeq, %cst, %cst_0 : f32
  %14 = arith.xori %12, %13 : i1
  %15 = arith.cmpf one, %cst, %cst_0 : f32
  %16 = arith.xori %14, %15 : i1
  return
})mlir";
  ASSERT_EQ(out, expected);
}

// check X+Y, where X is compile-time value (like 1) and Y is WrappedValue
TEST_F(ArithTest, EasyBuildFloatOperatorsLHSConst) {
  SKIP_IF_UNEXPECTED_FP_SIZE()
  auto out = composeFPIR(
      &context, [](EasyBuilder b) { return 33.0f; },
      [](EasyBuilder b) { return b(31.0f); });
  const char *expected =
      R"mlir(func.func @funcname() {
  %cst = arith.constant 3.100000e+01 : f32
  %cst_0 = arith.constant 3.300000e+01 : f32
  %0 = arith.addf %cst_0, %cst : f32
  %cst_1 = arith.constant 3.300000e+01 : f32
  %1 = arith.subf %cst_1, %cst : f32
  %cst_2 = arith.constant 3.300000e+01 : f32
  %2 = arith.mulf %cst_2, %cst : f32
  %cst_3 = arith.constant 3.300000e+01 : f32
  %3 = arith.divf %cst_3, %cst : f32
  %cst_4 = arith.constant 3.300000e+01 : f32
  %4 = arith.remf %cst_4, %cst : f32
  %cst_5 = arith.constant 3.300000e+01 : f32
  %5 = arith.cmpf olt, %cst_5, %cst : f32
  %cst_6 = arith.constant 3.300000e+01 : f32
  %6 = arith.cmpf ole, %cst_6, %cst : f32
  %7 = arith.andi %5, %6 : i1
  %cst_7 = arith.constant 3.300000e+01 : f32
  %8 = arith.cmpf ogt, %cst_7, %cst : f32
  %9 = arith.andi %7, %8 : i1
  %cst_8 = arith.constant 3.300000e+01 : f32
  %10 = arith.cmpf oge, %cst_8, %cst : f32
  %11 = arith.andi %9, %10 : i1
  %cst_9 = arith.constant 3.300000e+01 : f32
  %12 = arith.cmpf oeq, %cst_9, %cst : f32
  %13 = arith.xori %11, %12 : i1
  %cst_10 = arith.constant 3.300000e+01 : f32
  %14 = arith.cmpf one, %cst_10, %cst : f32
  %15 = arith.xori %13, %14 : i1
  return
})mlir";
  ASSERT_EQ(out, expected);
}

// check X+Y, where Y is compile-time value (like 1) and X is WrappedValue
TEST_F(ArithTest, EasyBuildFloatOperatorsRHSConst) {
  SKIP_IF_UNEXPECTED_FP_SIZE()
  auto out = composeFPIR(
      &context, [](EasyBuilder b) { return b(33.0f); },
      [](EasyBuilder b) { return 31.0f; });
  const char *expected =
      R"mlir(func.func @funcname() {
  %cst = arith.constant 3.300000e+01 : f32
  %cst_0 = arith.constant 3.100000e+01 : f32
  %0 = arith.addf %cst, %cst_0 : f32
  %cst_1 = arith.constant 3.100000e+01 : f32
  %1 = arith.subf %cst, %cst_1 : f32
  %cst_2 = arith.constant 3.100000e+01 : f32
  %2 = arith.mulf %cst, %cst_2 : f32
  %cst_3 = arith.constant 3.100000e+01 : f32
  %3 = arith.divf %cst, %cst_3 : f32
  %cst_4 = arith.constant 3.100000e+01 : f32
  %4 = arith.remf %cst, %cst_4 : f32
  %5 = arith.negf %cst : f32
  %cst_5 = arith.constant 3.100000e+01 : f32
  %6 = arith.cmpf olt, %cst, %cst_5 : f32
  %cst_6 = arith.constant 3.100000e+01 : f32
  %7 = arith.cmpf ole, %cst, %cst_6 : f32
  %8 = arith.andi %6, %7 : i1
  %cst_7 = arith.constant 3.100000e+01 : f32
  %9 = arith.cmpf ogt, %cst, %cst_7 : f32
  %10 = arith.andi %8, %9 : i1
  %cst_8 = arith.constant 3.100000e+01 : f32
  %11 = arith.cmpf oge, %cst, %cst_8 : f32
  %12 = arith.andi %10, %11 : i1
  %cst_9 = arith.constant 3.100000e+01 : f32
  %13 = arith.cmpf oeq, %cst, %cst_9 : f32
  %14 = arith.xori %12, %13 : i1
  %cst_10 = arith.constant 3.100000e+01 : f32
  %15 = arith.cmpf one, %cst, %cst_10 : f32
  %16 = arith.xori %14, %15 : i1
  return
})mlir";
  ASSERT_EQ(out, expected);
}

// check wrap<T>()
TEST_F(ArithTest, EasyBuildCheckWrap) {
  OpBuilder builder{&context};
  auto loc = builder.getUnknownLoc();
  EasyBuilder b{builder, loc};
  auto func = builder.create<func::FuncOp>(
      loc, "funcname",
      FunctionType::get(&context,
                        {MemRefType::get({100}, IntegerType::get(&context, 16)),
                         IntegerType::get(&context, 16)},
                        {}));

  builder.setInsertionPointToStart(func.addEntryBlock());
  // arg0 is of memref type
  auto arg0 = func.getArgument(0);
  EBValue wb = b(arg0); // check that it is ok to wrap generic value to EBValue
  auto expectedFail = b.wrapOrFail<EBFloatPoint>(arg0);
  ASSERT_TRUE(failed(expectedFail));

  auto arg1 = func.getArgument(1);
  auto expectedFail1 = b.wrapOrFail<EBFloatPoint>(arg1);
  ASSERT_TRUE(failed(expectedFail1));
  auto expectedOK1 = b.wrapOrFail<EBUnsigned>(arg1);
  ASSERT_TRUE(succeeded(expectedOK1));
  EBUnsigned u = b.wrap<EBUnsigned>(arg1);

  OpFoldResult foldresult = arg1;
  expectedFail1 = b.wrapOrFail<EBFloatPoint>(foldresult);
  ASSERT_TRUE(failed(expectedFail1));
  expectedOK1 = b.wrapOrFail<EBUnsigned>(foldresult);
  ASSERT_TRUE(succeeded(expectedOK1));
  u = b.wrap<EBUnsigned>(foldresult);

  foldresult = builder.getIndexAttr(123);
  expectedFail1 = b.wrapOrFail<EBFloatPoint>(foldresult);
  ASSERT_TRUE(failed(expectedFail1));
  expectedOK1 = b.wrapOrFail<EBUnsigned>(foldresult);
  ASSERT_TRUE(succeeded(expectedOK1));
  u = b.wrap<EBUnsigned>(foldresult);
}

TEST_F(ArithTest, EasyBuildCheckOpCall) {
  OpBuilder builder{&context};
  auto loc = builder.getUnknownLoc();
  EasyBuilder b{builder, loc};
  auto func = builder.create<func::FuncOp>(
      loc, "funcname",
      FunctionType::get(&context, {IntegerType::get(&context, 16)}, {}));

  builder.setInsertionPointToStart(func.addEntryBlock());

  auto arg0 = func.getArgument(0);
  auto v = b.wrap<EBUnsigned>(arg0);
  auto v2 = b.F<arith::MinUIOp, EBUnsigned>(v, b(uint16_t(1)));
  v2 = b.F<arith::MaxUIOp, EBUnsigned>(v2, b(uint16_t(100)));
  builder.create<func::ReturnOp>(loc);

  const char *expected =
      R"mlir(func.func @funcname(%arg0: i16) {
  %c1_i16 = arith.constant 1 : i16
  %0 = arith.minui %arg0, %c1_i16 : i16
  %c100_i16 = arith.constant 100 : i16
  %1 = arith.maxui %0, %c100_i16 : i16
  return
})mlir";
  std::string out;
  llvm::raw_string_ostream os{out};
  os << func;
  ASSERT_EQ(out, expected);
}
