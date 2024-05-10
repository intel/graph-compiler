//===- EasyBuildSCF.h - Easy IR Builder for general control flow *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the helper classes, functions and macros to help to
// build general structured control flow. Developers can use the utilities in
// this header to easily compose control flow IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_EASYBUILDSCF_H
#define MLIR_IR_EASYBUILDSCF_H
#include "gc/IR/EasyBuild.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
namespace easybuild {
namespace impl {

struct ForRangeSimulatorImpl {
  StatePtr s;
  LoopLikeOpInterface op;
  ForRangeSimulatorImpl(const StatePtr &s, LoopLikeOpInterface op)
      : s{s}, op{op} {
    s->builder.setInsertionPointToStart(&op.getLoopRegions().front()->front());
  }
  ~ForRangeSimulatorImpl() {
    s->builder.setInsertionPointAfter(op);
  }
};

template <int N, typename... Ts>
using NthTypeOf = typename std::tuple_element<N, std::tuple<Ts...>>::type;

template <typename... TArgs> struct ForVarBinding {
  ForRangeSimulatorImpl *impl;
  template <int I> auto get() {
    using TOut = NthTypeOf<I, TArgs...>;
    if (auto wrapped = TOut::wrapOrFail(
            impl->s, impl->op.getLoopRegions().front()->front().getArgument(I));
        succeeded(wrapped)) {
      return *wrapped;
    }
    llvm_unreachable("Bad cast for the loop iterator");
  }
};
} // namespace impl
} // namespace easybuild
} // namespace mlir

namespace std {
template <typename... TArgs>
struct tuple_size<mlir::easybuild::impl::ForVarBinding<TArgs...>>
    : std::integral_constant<std::size_t, sizeof...(TArgs)> {};

template <std::size_t I, typename... TArgs>
struct tuple_element<I, mlir::easybuild::impl::ForVarBinding<TArgs...>> {
  using type = mlir::easybuild::impl::NthTypeOf<I, TArgs...>;
};
} // namespace std

namespace mlir {
namespace easybuild {

namespace impl {

template <typename... TArgs> struct ForRangeSimulator : ForRangeSimulatorImpl {
  using ForRangeSimulatorImpl::ForRangeSimulatorImpl;
  struct ForRangeIterator {
    ForRangeSimulatorImpl *ptr;
    bool consumed;
    auto operator*() const { return ForVarBinding<TArgs...>{ptr}; }

    ForRangeIterator &operator++() {
      consumed = true;
      return *this;
    }

    bool operator!=(ForRangeIterator &other) const {
      return consumed != other.consumed;
    }

    ForRangeIterator(ForRangeSimulator *ptr)
        : ptr{ptr}, consumed{false} {}
    ForRangeIterator() : ptr{nullptr}, consumed{true} {}
  };

  ForRangeIterator begin() { return ForRangeIterator(this); }

  ForRangeIterator end() { return ForRangeIterator(); }
};
} // namespace impl

template <typename... TArgs>
auto forRangeIn(const impl::StatePtr &s, LoopLikeOpInterface op) {
  return impl::ForRangeSimulator<TArgs...>{s, op};
}

template <typename... TArgs>
auto forRangeIn(const EasyBuilder &s, LoopLikeOpInterface op) {
  return impl::ForRangeSimulator<TArgs...>{s.builder, op};
}

#define EB_for for

} // namespace easybuild
} // namespace mlir
#endif
