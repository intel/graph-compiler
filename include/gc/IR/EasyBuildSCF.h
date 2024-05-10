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
namespace scf {
class IfOp;
}

namespace easybuild {
namespace impl {

struct ForRangeSimulatorImpl {
  StatePtr s;
  LoopLikeOpInterface op;
  ForRangeSimulatorImpl(const StatePtr &s, LoopLikeOpInterface op)
      : s{s}, op{op} {
    s->builder.setInsertionPointToStart(&op.getLoopRegions().front()->front());
  }
  ~ForRangeSimulatorImpl() { s->builder.setInsertionPointAfter(op); }
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

    ForRangeIterator(ForRangeSimulator *ptr) : ptr{ptr}, consumed{false} {}
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

namespace impl {
struct IfSimulator;
struct IfIterator {
  IfSimulator *ptr;
  int index;
  int operator*() const;

  IfIterator &operator++() {
    index++;
    return *this;
  }

  bool operator!=(IfIterator &other) const { return index != other.index; }

  IfIterator(IfSimulator *ptr) : ptr{ptr}, index{0} {}
  IfIterator(int numRegions) : ptr{nullptr}, index{numRegions} {}
};

struct IfSimulator {
  StatePtr s;
  Operation *op;
  IfIterator begin() { return IfIterator(this); }
  IfIterator end() {
    int nonEmptyRegions = 0;
    for (auto &reg : op->getRegions()) {
      if (reg.begin() != reg.end()) {
        nonEmptyRegions++;
      }
    }
    return IfIterator(nonEmptyRegions);
  }
  ~IfSimulator() { s->builder.setInsertionPointAfter(op); }
};
inline int IfIterator::operator*() const {
  auto &blocks = ptr->op->getRegion(index);
  ptr->s->builder.setInsertionPointToStart(&blocks.back());
  return index;
}

} // namespace impl

impl::IfSimulator makeIfRange(const EasyBuilder &s, Operation *op) {
  return impl::IfSimulator{s.builder, op};
}

template <typename T = scf::IfOp>
impl::IfSimulator makeScfIfLikeRange(EBValue cond, TypeRange resultTypes) {
  auto &s = cond.builder;
  auto op = s->builder.create<T>(s->loc, resultTypes, cond, true);
  return impl::IfSimulator{s, op};
}

template <typename T = scf::IfOp>
impl::IfSimulator makeScfIfLikeRange(EBValue cond, bool hasElse = true) {
  auto &s = cond.builder;
  auto op = s->builder.create<T>(s->loc, TypeRange{}, cond, hasElse);
  return impl::IfSimulator{s, op};
}

#define EB_if(BUILDER, ...)                                                    \
  for (auto &&eb_mlir_if_scope__ :                                             \
       ::mlir::easybuild::makeIfRange(BUILDER, __VA_ARGS__))                   \
    if (eb_mlir_if_scope__ == 0)

// EB_scf_if(COND)
// EB_scf_if(COND, HAS_ELSE)
// EB_scf_if(COND, RESULT_TYPES)
#define EB_scf_if(...)                                                         \
  for (auto &&eb_mlir_if_scope__ :                                             \
       ::mlir::easybuild::makeScfIfLikeRange(__VA_ARGS__))                     \
    if (eb_mlir_if_scope__ == 0)
#define EB_else else

} // namespace easybuild
} // namespace mlir
#endif
