//===-- EasyBuild.h - DESC --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_IR_EASYBUILD_H
#define MLIR_IR_EASYBUILD_H
#include "mlir/IR/Builders.h"
#include <cstdint>
#include <memory>
#include <stddef.h>

namespace mlir {
namespace easybuild {

namespace impl {
struct EasyBuildState {
  OpBuilder &builder;
  Location loc;
  bool u64AsIndex;
  EasyBuildState(OpBuilder &builder, Location loc, bool u64AsIndex)
      : builder{builder}, loc{loc}, u64AsIndex{u64AsIndex} {}
};

using StatePtr = std::shared_ptr<impl::EasyBuildState>;

} // namespace impl

struct EBValue {
  std::shared_ptr<impl::EasyBuildState> builder;
  Value v;
  EBValue() = default;
  EBValue(const impl::StatePtr &builder, Value v) : builder{builder}, v{v} {}
  Value get() const { return v; }
  operator Value() const { return v; }

  static FailureOr<EBValue> wrapOrFail(const impl::StatePtr &state, Value v) {
    return EBValue{state, v};
  }
};

struct EBArithValue;

struct EasyBuilder {
  std::shared_ptr<impl::EasyBuildState> builder;
  EasyBuilder(OpBuilder &builder, Location loc, bool u64AsIndex = false)
      : builder{
            std::make_shared<impl::EasyBuildState>(builder, loc, u64AsIndex)} {}
  EasyBuilder(const std::shared_ptr<impl::EasyBuildState> &builder)
      : builder{builder} {}
  void setLoc(const Location &l) { builder->loc = l; }

  template <typename W, typename V> auto wrapOrFail(V &&v) {
    return W::wrapOrFail(builder, std::forward<V>(v));
  }

  Operation *getLastOperaion() {
    return &*(--builder->builder.getInsertionPoint());
  }

  template <typename W, typename V> auto wrap(V &&v) {
    auto ret = wrapOrFail<W>(std::forward<V>(v));
    if (failed(ret)) {
      llvm_unreachable("wrap failed!");
    }
    return *ret;
  }

  template <typename V> auto operator()(V &&v) {
    if constexpr (std::is_convertible_v<V, Value>) {
      return EBValue{builder, std::forward<V>(v)};
    } else {
      return wrap<EBArithValue>(std::forward<V>(v));
    }
  }

  template <typename W = EBArithValue> auto toIndex(uint64_t v) const {
    return W::toIndex(builder, v);
  }

  template <typename OP, typename OutT = EBValue, typename... Args>
  auto F(Args &&...v) {
    if constexpr (std::is_same_v<OutT, void>) {
      builder->builder.create<OP>(builder->loc, std::forward<Args>(v)...);
    } else {
      return wrap<OutT>(
          builder->builder.create<OP>(builder->loc, std::forward<Args>(v)...));
    }
  }

  template <typename OP = scf::YieldOp, typename... Args>
  auto yield(Args &&...v) {
    builder->builder.create<OP>(builder->loc,
                                ValueRange{std::forward<Args>(v)...});
  }
};

} // namespace easybuild
} // namespace mlir
#endif