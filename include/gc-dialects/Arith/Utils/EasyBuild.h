//===- EasyBuild.h - Easy Arith IR Builder utilities ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// the Arith dialect. These are not passes by themselves but are used
// either by passes, optimization sequences, or in turn by other transformation
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef GC_DIALECT_ARITH_UTILS_EASYBUILD_H
#define GC_DIALECT_ARITH_UTILS_EASYBUILD_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/EasyBuild.h"
#include <cstdint>
#include <memory>
#include <stddef.h>

namespace mlir {
namespace easybuild {

namespace impl {

template <std::size_t size>
struct ToFloatType {};

template <>
struct ToFloatType<4> {
  using type = Float32Type;
};
template <>
struct ToFloatType<8> {
  using type = Float64Type;
};

inline Type getElementType(Value v) {
  auto type = v.getType();
  if (type.isa<TensorType>() || type.isa<VectorType>()) {
    type = type.cast<ShapedType>().getElementType();
  }
  return type;
}

} // namespace impl

struct EBUnsigned;

struct EBArithValue : public EBValue {
  template <typename T = EBUnsigned>
  static T toIndex(const impl::StatePtr &state, uint64_t v);

  template <typename T>
  static auto wrapOrFail(const impl::StatePtr &state, T &&v);

  template <typename T>
  static auto wrap(const impl::StatePtr &state, T &&v) {
    auto ret = wrapOrFail<T>(state, std::forward<T>(v));
    if (failed(ret)) {
      llvm_unreachable("Bad wrap");
    }
    return *ret;
  }

protected:
  using EBValue::EBValue;
};

struct EBUnsigned : public EBArithValue {
  static FailureOr<EBUnsigned> wrapOrFail(const impl::StatePtr &state,
                                          Value v) {
    auto type = impl::getElementType(v);
    if (type.isUnsignedInteger() || type.isSignlessInteger() ||
        type.isIndex()) {
      return EBUnsigned{state, v};
    }
    return failure();
  }
  static FailureOr<EBUnsigned> wrapOrFail(const impl::StatePtr &state,
                                          const OpFoldResult &v) {
    if (v.is<Value>()) {
      return wrapOrFail(state, v.get<Value>());
    }
    auto attr = v.get<Attribute>();
    if (auto val = attr.dyn_cast<IntegerAttr>()) {
      if (val.getType().isIndex())
        return EBUnsigned{state, state->builder.create<arith::ConstantIndexOp>(
                                     state->loc, val.getInt())};
      else
        return EBUnsigned{state, state->builder.create<arith::ConstantIntOp>(
                                     state->loc, val.getInt(), val.getType())};
    }
    return failure();
  }
  friend struct EBArithValue;
  friend struct OperatorHandlers;

protected:
  using EBArithValue::EBArithValue;
};

struct EBSigned : EBArithValue {
  static FailureOr<EBSigned> wrapOrFail(const impl::StatePtr &state, Value v) {
    auto type = impl::getElementType(v);
    if (type.isSignedInteger() || type.isSignlessInteger()) {
      return EBSigned{state, v};
    }
    return failure();
  }
  static FailureOr<EBSigned> wrapOrFail(const impl::StatePtr &state,
                                        const OpFoldResult &v) {
    if (v.is<Value>()) {
      return wrapOrFail(state, v.get<Value>());
    }
    auto attr = v.get<Attribute>();
    if (auto val = attr.dyn_cast<IntegerAttr>())
      return EBSigned{state, state->builder.create<arith::ConstantIntOp>(
                                 state->loc, val.getInt(), val.getType())};
    return failure();
  }
  friend struct EBArithValue;
  friend struct OperatorHandlers;

protected:
  using EBArithValue::EBArithValue;
};

struct EBFloatPoint : EBArithValue {
  static FailureOr<EBFloatPoint> wrapOrFail(const impl::StatePtr &state,
                                            Value v) {
    auto type = impl::getElementType(v);
    if (type.isa<FloatType>()) {
      return EBFloatPoint{state, v};
    }
    return failure();
  }
  static FailureOr<EBFloatPoint> wrapOrFail(const impl::StatePtr &state,
                                            const OpFoldResult &v) {
    if (v.is<Value>()) {
      return wrapOrFail(state, v.get<Value>());
    }
    auto attr = v.get<Attribute>();
    if (auto val = attr.dyn_cast<FloatAttr>())
      return EBFloatPoint{state, state->builder.create<arith::ConstantFloatOp>(
                                     state->loc, val.getValue(),
                                     val.getType().cast<FloatType>())};
    return failure();
  }
  friend struct EBArithValue;
  friend struct OperatorHandlers;

protected:
  using EBArithValue::EBArithValue;
};

template <typename T>
inline T EBArithValue::toIndex(const impl::StatePtr &state, uint64_t v) {
  return EBUnsigned{
      state, state->builder.create<arith::ConstantIndexOp>(state->loc, v)};
}

template <typename T>
inline auto EBArithValue::wrapOrFail(const impl::StatePtr &state, T &&v) {
  using DT = std::decay_t<T>;
  static_assert(std::is_arithmetic_v<DT>, "Expecting arithmetic types");
  if constexpr (std::is_same_v<DT, uint64_t>) {
    if (state->u64AsIndex) {
      return FailureOr<EBUnsigned>{toIndex(state, v)};
    }
  }

  if constexpr (std::is_same_v<DT, bool>) {
    return FailureOr<EBUnsigned>{
        EBUnsigned{state, state->builder.create<arith::ConstantIntOp>(
                              state->loc, static_cast<int64_t>(v), 1)}};
  } else if constexpr (std::is_integral_v<DT>) {
    if constexpr (!std::is_signed_v<DT>) {
      return FailureOr<EBUnsigned>{EBUnsigned{
          state, state->builder.create<arith::ConstantIntOp>(
                     state->loc, static_cast<int64_t>(v), sizeof(T) * 8)}};
    } else {
      return FailureOr<EBSigned>{EBSigned{
          state, state->builder.create<arith::ConstantIntOp>(
                     state->loc, static_cast<int64_t>(v), sizeof(T) * 8)}};
    }
  } else {
    using DType = typename impl::ToFloatType<sizeof(DT)>::type;
    return FailureOr<EBFloatPoint>{
        EBFloatPoint{state, state->builder.create<arith::ConstantFloatOp>(
                                state->loc, APFloat{v},
                                DType::get(state->builder.getContext()))}};
  }
}

struct OperatorHandlers {
  template <typename OP, typename V>
  static V handleBinary(const V &a, const V &b) {
    assert(a.builder == b.builder);
    return {a.builder,
            a.builder->builder.template create<OP>(a.builder->loc, a.v, b.v)};
  }

  template <typename OP, typename V, typename T2>
  static V handleBinaryConst(const V &a, const T2 &b) {
    return handleBinary<OP>(a, EBArithValue::wrap(a.builder, b));
  }

  template <typename OP, typename V, typename T2>
  static V handleBinaryConst(const T2 &a, const V &b) {
    return handleBinary<OP>(EBArithValue::wrap(b.builder, a), b);
  }

  template <typename OP, typename V, typename Pred>
  static EBUnsigned handleCmp(const V &a, const V &b, Pred predicate) {
    assert(a.builder == b.builder);
    return {a.builder, a.builder->builder.template create<OP>(
                           a.builder->loc, predicate, a.v, b.v)};
  }

  template <typename OP, typename V, typename T2, typename Pred>
  static EBUnsigned handleCmpConst(const V &a, const T2 &b, Pred predicate) {
    return handleCmp<OP>(a, EBArithValue::wrap(a.builder, b), predicate);
  }

  template <typename OP, typename V, typename T2, typename Pred>
  static EBUnsigned handleCmpConst(const T2 &a, const V &b, Pred predicate) {
    return handleCmp<OP>(EBArithValue::wrap(b.builder, a), b, predicate);
  }

  template <typename T, typename TOp, typename... Args>
  static T create(const impl::StatePtr &state, Args &&...v) {
    return {state,
            state->builder.create<TOp>(state->loc, std::forward<Args>(v)...)};
  }
};

#define DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, OPCLASS, TYPE)              \
  inline TYPE operator OP(const TYPE &a, const TYPE &b) {                      \
    return OperatorHandlers::handleBinary<OPCLASS>(a, b);                      \
  }                                                                            \
  template <typename T>                                                        \
  inline TYPE operator OP(const TYPE &a, T b) {                                \
    return OperatorHandlers::handleBinaryConst<OPCLASS, TYPE>(a, b);           \
  }                                                                            \
  template <typename T>                                                        \
  inline TYPE operator OP(T a, const TYPE &b) {                                \
    return OperatorHandlers::handleBinaryConst<OPCLASS, TYPE>(a, b);           \
  }

#define DEF_EASYBUILD_BINARY_OPERATOR(OP, SIGNED, UNSIGNED, FLOAT)             \
  DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, SIGNED, EBSigned)                 \
  DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, UNSIGNED, EBUnsigned)             \
  DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, FLOAT, EBFloatPoint)

DEF_EASYBUILD_BINARY_OPERATOR(+, arith::AddIOp, arith::AddIOp, arith::AddFOp)
DEF_EASYBUILD_BINARY_OPERATOR(-, arith::SubIOp, arith::SubIOp, arith::SubFOp)
DEF_EASYBUILD_BINARY_OPERATOR(*, arith::MulIOp, arith::MulIOp, arith::MulFOp)
DEF_EASYBUILD_BINARY_OPERATOR(/, arith::DivSIOp, arith::DivUIOp, arith::DivFOp)
DEF_EASYBUILD_BINARY_OPERATOR(%, arith::RemSIOp, arith::RemUIOp, arith::RemFOp)

#undef DEF_EASYBUILD_BINARY_OPERATOR
#define DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(OP, SIGNED, UNSIGNED)            \
  DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, SIGNED, EBSigned)                 \
  DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, UNSIGNED, EBUnsigned)

DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(>>, arith::ShRSIOp, arith::ShRUIOp)
DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(<<, arith::ShLIOp, arith::ShLIOp)
DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(&, arith::AndIOp, arith::AndIOp)
DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(|, arith::OrIOp, arith::OrIOp)
DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(^, arith::XOrIOp, arith::XOrIOp)

#undef DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT
#undef DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE

inline EBFloatPoint operator-(const EBFloatPoint &a) {
  return OperatorHandlers::create<EBFloatPoint, arith::NegFOp>(a.builder, a.v);
}

#define DEF_EASYBUILD_CMP_OPERATOR(OP, OPCLASS, TYPE, PRED)                    \
  EBUnsigned operator OP(const TYPE &a, const TYPE &b) {                       \
    return OperatorHandlers::handleCmp<OPCLASS>(a, b, PRED);                   \
  }                                                                            \
  template <typename T>                                                        \
  EBUnsigned operator OP(const TYPE &a, T b) {                                 \
    return OperatorHandlers::handleCmpConst<OPCLASS, TYPE>(a, b, PRED);        \
  }                                                                            \
  template <typename T>                                                        \
  EBUnsigned operator OP(T a, const TYPE &b) {                                 \
    return OperatorHandlers::handleCmpConst<OPCLASS, TYPE>(a, b, PRED);        \
  }

DEF_EASYBUILD_CMP_OPERATOR(<, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::ult)
DEF_EASYBUILD_CMP_OPERATOR(<=, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::ule)
DEF_EASYBUILD_CMP_OPERATOR(>, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::ugt)
DEF_EASYBUILD_CMP_OPERATOR(>=, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::uge)
DEF_EASYBUILD_CMP_OPERATOR(==, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::eq)
DEF_EASYBUILD_CMP_OPERATOR(!=, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::ne)

DEF_EASYBUILD_CMP_OPERATOR(<, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::slt)
DEF_EASYBUILD_CMP_OPERATOR(<=, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::sle)
DEF_EASYBUILD_CMP_OPERATOR(>, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::sgt)
DEF_EASYBUILD_CMP_OPERATOR(>=, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::sge)
DEF_EASYBUILD_CMP_OPERATOR(==, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::eq)
DEF_EASYBUILD_CMP_OPERATOR(!=, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::ne)

DEF_EASYBUILD_CMP_OPERATOR(<, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::OLT)
DEF_EASYBUILD_CMP_OPERATOR(<=, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::OLE)
DEF_EASYBUILD_CMP_OPERATOR(>, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::OGT)
DEF_EASYBUILD_CMP_OPERATOR(>=, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::OGE)
DEF_EASYBUILD_CMP_OPERATOR(==, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::OEQ)
DEF_EASYBUILD_CMP_OPERATOR(!=, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::ONE)

#undef DEF_EASYBUILD_CMP_OPERATOR

namespace arithops {
inline EBFloatPoint castIntToFP(Type type, const EBSigned &v) {
  return OperatorHandlers::create<EBFloatPoint, arith::SIToFPOp>(v.builder,
                                                                 type, v);
}

inline EBFloatPoint castIntToFP(Type type, const EBUnsigned &v) {
  return OperatorHandlers::create<EBFloatPoint, arith::UIToFPOp>(v.builder,
                                                                 type, v);
}

template <typename T>
inline T castFPToInt(const EBFloatPoint &v) {
  if constexpr (std::is_same_v<T, EBSigned>) {
    return OperatorHandlers::create<EBSigned, arith::FPToSIOp>(v.builder, v);
  } else {
    static_assert(std::is_same_v<T, EBUnsigned>,
                  "Expecting EBUnsigned or EBSigned");
    return OperatorHandlers::create<EBUnsigned, arith::FPToUIOp>(v.builder, v);
  }
}

inline EBSigned ceildiv(const EBSigned &a, const EBSigned &b) {
  return OperatorHandlers::create<EBSigned, arith::CeilDivSIOp>(a.builder, a,
                                                                b);
}

inline EBUnsigned ceildiv(const EBUnsigned &a, const EBUnsigned &b) {
  return OperatorHandlers::create<EBUnsigned, arith::CeilDivUIOp>(a.builder, a,
                                                                  b);
}

inline EBSigned floordiv(const EBSigned &a, const EBSigned &b) {
  return OperatorHandlers::create<EBSigned, arith::FloorDivSIOp>(a.builder, a,
                                                                 b);
}

inline EBSigned extend(Type type, const EBSigned &a) {
  return OperatorHandlers::create<EBSigned, arith::ExtSIOp>(a.builder, type, a);
}

inline EBUnsigned extend(Type type, const EBUnsigned &a) {
  return OperatorHandlers::create<EBUnsigned, arith::ExtUIOp>(a.builder, type,
                                                              a);
}

inline EBFloatPoint extend(Type type, const EBFloatPoint &a) {
  return OperatorHandlers::create<EBFloatPoint, arith::ExtFOp>(a.builder, type,
                                                               a);
}

inline EBSigned trunc(Type type, const EBSigned &a) {
  return OperatorHandlers::create<EBSigned, arith::TruncIOp>(a.builder, type,
                                                             a);
}

inline EBFloatPoint trunc(Type type, const EBFloatPoint &a) {
  return OperatorHandlers::create<EBFloatPoint, arith::TruncFOp>(a.builder,
                                                                 type, a);
}

template <typename T>
inline T select(const EBUnsigned &pred, const T &trueValue,
                const T &falseValue) {
  static_assert(std::is_base_of_v<EBArithValue, T>,
                "Expecting T to be a subclass of EBArithValue");
  return OperatorHandlers::create<T, arith::SelectOp>(pred.builder, pred,
                                                      trueValue, falseValue);
}

template <typename TyTo, typename TyFrom>
inline TyTo bitcast(Type type, const TyFrom &v) {
  return OperatorHandlers::create<TyTo, arith::BitcastOp>(v.builder, type, v);
}

inline EBSigned min(const EBSigned &a, const EBSigned &b) {
  return OperatorHandlers::create<EBSigned, arith::MinSIOp>(a.builder, a, b);
}

inline EBSigned max(const EBSigned &a, const EBSigned &b) {
  return OperatorHandlers::create<EBSigned, arith::MaxSIOp>(a.builder, a, b);
}

inline EBUnsigned min(const EBUnsigned &a, const EBUnsigned &b) {
  return OperatorHandlers::create<EBUnsigned, arith::MinUIOp>(a.builder, a, b);
}

inline EBUnsigned max(const EBUnsigned &a, const EBUnsigned &b) {
  return OperatorHandlers::create<EBUnsigned, arith::MaxUIOp>(a.builder, a, b);
}

inline EBFloatPoint minnum(const EBFloatPoint &a, const EBFloatPoint &b) {
  return OperatorHandlers::create<EBFloatPoint, arith::MinNumFOp>(a.builder, a,
                                                                  b);
}

inline EBFloatPoint maxnum(const EBFloatPoint &a, const EBFloatPoint &b) {
  return OperatorHandlers::create<EBFloatPoint, arith::MaxNumFOp>(a.builder, a,
                                                                  b);
}

inline EBFloatPoint minimum(const EBFloatPoint &a, const EBFloatPoint &b) {
  return OperatorHandlers::create<EBFloatPoint, arith::MinimumFOp>(a.builder, a,
                                                                   b);
}

inline EBFloatPoint maximum(const EBFloatPoint &a, const EBFloatPoint &b) {
  return OperatorHandlers::create<EBFloatPoint, arith::MaximumFOp>(a.builder, a,
                                                                   b);
}

} // namespace arithops

} // namespace easybuild
} // namespace mlir
#endif
