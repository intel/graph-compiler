//===-- Transform.h - Transformation untils ----------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_TRANSFORM_H
#define GC_TRANSFORM_H

#include <variant>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>

#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/XeGPU/uArch/uArchCommon.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "gc/Runtime/GpuRuntime.h"

namespace mlir::gc {

// ------------------- Attribute utilities ------------------ //
constexpr char GC_ATTR_KERNEL_NAME[] = "gc.kernel_name";
constexpr char GC_ATTR_LEVEL[] = "gc.tiling.level";
constexpr char GC_ATTR_WG_TILE_SIZES[] = "gc.tiling.wg_tile_sizes";

template <typename T> auto createAttr(MLIRContext *ctx, T value) {
  if constexpr (std::is_integral_v<T>) {
    auto type = IntegerType::get(ctx, sizeof(T) * 8);
    return IntegerAttr::get(type, static_cast<int64_t>(value));
  } else if constexpr (std::is_enum_v<T>) {
    return createAttr(ctx, static_cast<std::underlying_type_t<T>>(value));
  } else if constexpr (std::is_floating_point_v<T>) {
    Type type;
    if constexpr (sizeof(T) == 4) {
      type = Float32Type::get(ctx);
    } else if constexpr (sizeof(T) == 8) {
      type = Float64Type::get(ctx);
    }
    return FloatAttr::get(type, static_cast<double>(value));
  } else if constexpr (std::is_convertible_v<T, Attribute>) {
    return value;
  } else if constexpr (std::is_convertible_v<T, StringRef>) {
    return StringAttr::get(ctx, value);
  } else if constexpr (std::is_convertible_v<
                           T, ArrayRef<typename T::value_type>>) {
    SmallVector<Attribute> attrs;
    for (const auto &v : value) {
      attrs.push_back(createAttr(ctx, v));
    }
    return ArrayAttr::get(ctx, attrs);
  }
}

template <typename T> auto getAttrValue(Attribute attr) {
  if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(cast<IntegerAttr>(attr).getInt());
  } else if constexpr (std::is_enum_v<T>) {
    return static_cast<T>(getAttrValue<std::underlying_type_t<T>>(attr));
  } else if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(cast<FloatAttr>(attr).getValueAsDouble());
  } else if constexpr (std::is_convertible_v<T, Attribute>) {
    return cast<T>(attr);
  } else if constexpr (std::is_convertible_v<T, StringRef>) {
    return cast<StringAttr>(attr).getValue().data();
  } else if constexpr (std::is_convertible_v<
                           T, ArrayRef<typename T::value_type>>) {
    using ElemTy = typename T::value_type;
    SmallVector<ElemTy> values;
    auto arrayAttr = cast<ArrayAttr>(attr);
    for (auto elemAttr : arrayAttr.getValue()) {
      values.push_back(getAttrValue<ElemTy>(elemAttr));
    }
    return values;
  }
}

template <typename T>
auto getDiscardableAttr(Operation *op, StringRef name, T defaultValue) {
  auto attr = op->getDiscardableAttr(name);
  return attr ? getAttrValue<T>(attr) : defaultValue;
}

template <typename... Path> struct GcAttrs {
  GcAttrs(Operation *op, Path... p) : path(p...), op(op), attrs(nullptr) {}
  GcAttrs(const GcAttrs &) = default;
  GcAttrs(GcAttrs &&) = default;
  ~GcAttrs() { save(); }

  GcAttrs &operator=(GcAttrs &&other) {
    save();
    path = std::move(other.path);
    op = other.op;
    attrs = std::move(other.attrs);
    return *this;
  }

  Attribute getAttr(StringRef name) {
    load();
    if (std::holds_alternative<DictionaryAttr>(attrs)) {
      auto &attr = std::get<DictionaryAttr>(attrs);
      return attr ? attr.get(name) : nullptr;
    }
    return std::get<NamedAttrList>(attrs).get(name);
  }

  template <typename T> std::optional<T> get(StringRef name) {
    auto attr = getAttr(name);
    return attr ? std::optional<T>(getAttrValue<T>(attr)) : std::nullopt;
  }

  template <typename T> T get(StringRef name, T defaultValue) {
    auto attr = getAttr(name);
    return attr ? getAttrValue<T>(attr) : defaultValue;
  }

  template <typename T> void set(StringRef name, T value) {
    auto attr = createAttr(mod()->getContext(), value);
    if (!std::holds_alternative<NamedAttrList>(attrs)) {
      load();
      attrs = NamedAttrList(std::get<DictionaryAttr>(attrs));
    }
    std::get<NamedAttrList>(attrs).set(name, attr);
  }

  bool exists() {
    load();
    return std::holds_alternative<DictionaryAttr>(attrs) &&
           static_cast<bool>(std::get<DictionaryAttr>(attrs));
  }

  void save() {
    if (std::holds_alternative<NamedAttrList>(attrs)) {
      auto op = mod();
      std::apply(
          [&](auto... p) {
            NamedAttrList list(dyn_cast_if_present<DictionaryAttr>(
                op->getDiscardableAttr(ROOT)));
            save(op, list, std::get<NamedAttrList>(attrs), p...);
            if (list.empty()) op->removeDiscardableAttr(ROOT);
            else op->setDiscardableAttr(ROOT, toDict(op, list));
          },
          path);
      attrs = toDict(op, std::get<NamedAttrList>(attrs));
    }
  }

private:
  static constexpr char ROOT[] = "gc.module";
  std::tuple<Path...> path;
  mutable Operation *op;
  mutable std::variant<std::nullptr_t, DictionaryAttr, NamedAttrList> attrs;

  Operation *mod() {
    if (!isa<ModuleOp>(op)) op = op->getParentOfType<ModuleOp>();
    return op;
  }

  void load() { // lazy load
    if (std::holds_alternative<std::nullptr_t>(attrs)) {
      std::apply([&](auto... p) { attrs = load(mod(), p...); }, path);
    }
  }

  template <typename... P> static DictionaryAttr load(Operation *op, P... p) {
    auto attr =
        dyn_cast_if_present<DictionaryAttr>(op->getDiscardableAttr(ROOT));
    if (attr) {
      ((attr = attr ? dyn_cast_if_present<DictionaryAttr>(attr.get(p)) : attr),
       ...);
    }
    return attr;
  }

  template <typename T>
  static void save(Operation *op, NamedAttrList &list,
                   const NamedAttrList &newValues, T name) {
    if (newValues.empty()) list.erase(name);
    else list.set(name, toDict(op, newValues));
  }

  template <typename T, typename... P>
  static void save(Operation *op, NamedAttrList &list,
                   const NamedAttrList &newValues, T name, P... path) {
    NamedAttrList newList(dyn_cast_if_present<DictionaryAttr>(list.get(name)));
    save(op, newList, newValues, path...);
    if (newList.empty()) list.erase(name);
    else list.set(name, toDict(op, newList));
  }

  static DictionaryAttr toDict(Operation *op, const NamedAttrList &list) {
    return list.getDictionary(op->getContext());
  }
};

struct DevAttrs : public GcAttrs<const char *> {
  DevAttrs(Operation *op) : GcAttrs<const char *>(op, "device") {}

  std::optional<uint32_t> getId() { return get<uint32_t>(ID); }
  void setId(uint32_t id) { set(ID, id); }

  std::optional<StringRef> getName() { return get<StringRef>(NAME); }
  void setName(StringRef name) { set(NAME, name); }

  std::optional<StringRef> getArch() { return get<StringRef>(DEVICE_ARCH); }
  void setArch(StringRef arch) {
    set(DEVICE_ARCH, arch);
    this->arch = nullptr;
  }

  const xegpu::uArch::uArch *getUarch() {
    if (!arch) {
      arch = xegpu::uArch::getUArch(getArch().value_or("bmg"));
    }
    return arch;
  }

  std::optional<SmallVector<size_t>> getSgSizes() {
    return get<SmallVector<size_t>>(SG_SIZES);
  }
  void setSgSizes(ArrayRef<size_t> sizes) { set(SG_SIZES, sizes); }

  std::optional<size_t> getMaxSgSize() {
    if (auto sgSizes = getSgSizes(); sgSizes && !sgSizes->empty()) {
      return *llvm::max_element(*sgSizes);
    }
    return std::nullopt;
  }

  std::optional<size_t> getMaxWgSize() { return get<size_t>(MAX_WG_SIZE); }
  void setMaxWgSize(size_t size) { set(MAX_WG_SIZE, size); }

  static const std::optional<const char *> getDeviceArch(int id) {
    auto arch = ::gc::gpu::DeviceInfo::idToArch(id);
    return arch[0] == '\0' ? std::nullopt : std::optional<const char *>(arch);
  }

private:
  static constexpr char ID[] = "id";
  static constexpr char NAME[] = "name";
  static constexpr char DEVICE_ARCH[] = "arch";
  static constexpr char MAX_WG_SIZE[] = "max_wg_size";
  static constexpr char SG_SIZES[] = "sg_sizes";
  const xegpu::uArch::uArch *arch = nullptr;
};

struct KernelAttrs : public GcAttrs<const char *, StringRef> {
  KernelAttrs(Operation *op, StringRef name)
      : GcAttrs<const char *, StringRef>(op, "kernels", name) {}

  std::optional<SmallVector<size_t>> getTiles() {
    return get<SmallVector<size_t>>(TILES);
  }
  void setTiles(ArrayRef<size_t> tiles) { set(TILES, tiles); }

  std::optional<SmallVector<size_t>> getThreads() {
    return get<SmallVector<size_t>>(THREADS);
  }
  void setThreads(ArrayRef<size_t> threads) { set(THREADS, threads); }

  // Number of subgroups per non-reduction dim, i.e. wgTile/sgTile.
  std::optional<SmallVector<size_t>> getSgLayout() {
    return get<SmallVector<size_t>>(SG_LAYOUT);
  }
  void setSgLayout(ArrayRef<size_t> layout) { set(SG_LAYOUT, layout); }

  // Mapping of the kernel to the main function arguments
  std::optional<SmallVector<size_t>> getArgsMap() {
    return get<SmallVector<size_t>>(ARGS_MAP);
  }
  void setArgsMap(ArrayRef<size_t> argsMap) { set(ARGS_MAP, argsMap); }

  // A flattened GpuDim is [mul, div, nrefs, arg0, dim0, ...]. arg_sizes[i]
  // is [rank, <GpuDim>...], while blocks contains exactly three GpuDims.
  std::optional<SmallVector<SmallVector<int64_t>>> getArgSizes() {
    return get<SmallVector<SmallVector<int64_t>>>(ARG_SIZES);
  }
  void setArgSizes(ArrayRef<SmallVector<int64_t>> argSizes) {
    set(ARG_SIZES, argSizes);
  }

  std::optional<SmallVector<SmallVector<int64_t>>> getBlocks() {
    return get<SmallVector<SmallVector<int64_t>>>(BLOCKS);
  }
  void setBlocks(ArrayRef<SmallVector<int64_t>> blocks) { set(BLOCKS, blocks); }

  std::optional<SmallVector<StringRef>> getDepends() {
    return get<SmallVector<StringRef>>(DEPENDS);
  }
  void setDepends(ArrayRef<StringRef> depends) { set(DEPENDS, depends); }

  template <typename T = size_t> std::optional<T> getWgSize() {
    return get<T>(WG_SIZE);
  }
  template <typename T = size_t> void setWgSize(T wgSize) {
    set(WG_SIZE, static_cast<T>(wgSize));
  }

  template <typename T = size_t> std::optional<T> getSgSize() {
    return get<T>(SG_SIZE);
  }
  template <typename T = size_t> void setSgSize(T sgSize) {
    set(SG_SIZE, static_cast<T>(sgSize));
  }

private:
  static constexpr char TILES[] = "tiles";
  static constexpr char THREADS[] = "threads";
  static constexpr char WG_SIZE[] = "wg_size";
  static constexpr char SG_SIZE[] = "sg_size";
  static constexpr char SG_LAYOUT[] = "sg_layout";
  static constexpr char DEPENDS[] = "depends";
  static constexpr char ARGS_MAP[] = "args_map";
  static constexpr char ARG_SIZES[] = "arg_sizes";
  static constexpr char BLOCKS[] = "blocks";
};
// ---------------------------------------------------------- //

// This class is a placeholder for the rewriter-related boilerplate code.
struct OpRewriter final : IRRewriter {
  Location loc;

  explicit OpRewriter(func::FuncOp &func)
      : IRRewriter(func.getContext()), loc(func.getLoc()) {}

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    return OpTy::create(*this, loc, std::forward<Args>(args)...);
  }

  arith::ConstantIndexOp createConstant(int64_t v) {
    return create<arith::ConstantIndexOp>(v);
  }

  arith::ConstantFloatOp createConstant(double v) {
    return create<arith::ConstantFloatOp>(getF64Type(), APFloat(v));
  }
};

struct TruePredicate {
  template <typename T> constexpr bool operator()(T &&) const { return true; }
};
template <typename Predicate = TruePredicate> struct ReverseIterator {

  static auto makeIterable(Block &block) {
    auto reversed = llvm::reverse(ForwardIterator::makeIterable(block));
    if constexpr (std::is_same_v<Predicate, TruePredicate>) return reversed;
    else return llvm::make_filter_range(reversed, Predicate{});
  }

  template <typename T> static auto makeIterable(T &range) {
    return llvm::reverse(ForwardIterator::makeIterable(range));
  }
};

template <typename T = Operation *, typename Predicate = TruePredicate>
T findLast(Operation *root, std::function<bool(T)> predicate) {
  T last = nullptr;
  root->walk<WalkOrder::PreOrder, ReverseIterator<Predicate>>([&](T op) {
    if (predicate(op)) {
      last = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return last;
}
// ---------------------------------------------------------- //

// Check recursively if the specified operation has an operand that
// depends on a result of a previous operation, matching the predicate.
template <unsigned MaxDepth = std::numeric_limits<unsigned>::max()>
bool isOperandDependsOnOp(std::function<bool(Operation *)> predicate,
                          Operation *operation, unsigned depth = 0) {
  for (auto operand : operation->getOperands()) {
    if (auto op = operand.getDefiningOp();
        op &&
        (predicate(op) || (depth < MaxDepth &&
                           isOperandDependsOnOp(predicate, op, depth + 1)))) {
      return true;
    }
  }
  return false;
}

// Check recursively if there are any operation, matching the predicate, that
// depends on the result of the specified operation.
template <unsigned MaxDepth = std::numeric_limits<unsigned>::max()>
bool isOpDependsOnResult(std::function<bool(Operation *)> predicate,
                         Operation *operation, unsigned depth = 0) {
  for (auto res : operation->getResults()) {
    for (auto u : res.getUsers()) {
      if (predicate(u) ||
          (depth < MaxDepth && isOpDependsOnResult(predicate, u, depth + 1))) {
        return true;
      }
    }
  }
  return false;
}

inline bool isMatmulOp(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  return linalgOp && linalg::isaContractionOpInterface(linalgOp);
  // TODO: Check matmul like generics
}

inline bool containsMatmulOp(Operation *op) {
  if (isMatmulOp(op)) return true;
  bool found = false;
  op->walk([&](Operation *nested) {
    if (nested != op && isMatmulOp(nested)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// If a slice inside the loop is created from an external empty tensor and
// the tensor is not passed to the loop's shared_outs, but referenced
// directly, replace the slice with an empty tensor of the same size.
inline void replaceEmptySlices(OpRewriter &rw, LoopLikeOpInterface loop) {
  loop.walk([&](tensor::ExtractSliceOp slice) {
    if (auto empty = slice.getSource().getDefiningOp<tensor::EmptyOp>();
        empty && empty->getParentOfType<LoopLikeOpInterface>() != loop) {
      auto type = slice.getType();
      rw.setInsertionPointAfter(slice);
      SmallVector<Value> dynDims;
      for (int64_t i = 0, r = type.getRank(); i < r; ++i) {
        if (type.isDynamicDim(i)) {
          dynDims.push_back(rw.create<tensor::DimOp>(slice, i));
        }
      }
      rw.replaceOp(slice, rw.create<tensor::EmptyOp>(
                              type.getShape(), type.getElementType(), dynDims));
    }
  });
}

inline scf::ForallOp getKernelLoop(Operation *op) {
  for (; op; op = op->getParentOp()) {
    if (auto loop = dyn_cast<scf::ForallOp>(op);
        loop && loop->hasAttr(GC_ATTR_KERNEL_NAME))
      return loop;
  }
  return nullptr;
}

inline void canonicalizeLoop(LoopLikeOpInterface &loop) {
  auto parent = loop->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
  assert(parent);
  auto ctx = parent->getContext();
  RewritePatternSet patterns(ctx);
  if (isa<scf::ForallOp>(loop.getOperation())) {
    scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
  } else if (isa<scf::ForOp>(loop.getOperation())) {
    scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  }

  constexpr char stampAttrName[] = "gc.loop.stamp";
  static size_t stamp = 0;
  auto st = ++stamp;
  // The loop's operation can be replaced by the patterns. Using a stamp
  // to find it again.
  loop->setDiscardableAttr(stampAttrName, createAttr(ctx, st));
  if (failed(applyPatternsGreedily(parent, std::move(patterns)))) {
    loop->emitWarning() << "Loop canonicalization failed";
  }
  parent->walk([&](LoopLikeOpInterface o) {
    if (getDiscardableAttr<size_t>(o, stampAttrName, 0) == st) {
      o->removeDiscardableAttr(stampAttrName);
      loop = o;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

// `result` is an OpResult produced by some srcOp.
// `consumer` is a consumer of this result.
// Check if the `consumer` has any other input, that depend on `srcOp` through a
// different path.
inline bool hasDiamondDep(OpResult result, Operation *consumer) {
  auto srcOp = result.getDefiningOp();
  SmallVector<Operation *> stack = llvm::filter_to_vector(
      llvm::map_range(consumer->getOperands(),
                      [&](Value v) { return v.getDefiningOp(); }),
      [&](Operation *o) { return o && o != srcOp; });
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!stack.empty()) {
    Operation *op = stack.pop_back_val();
    if (!visited.insert(op).second) continue;
    if (op == srcOp) return true;
    for (auto operand : op->getOperands()) {
      if (auto defOp = operand.getDefiningOp()) stack.push_back(defOp);
    }
  }
  return false;
}

// Check if all subgraphs from the `result` have a common intersection.
template <typename Predicate>
bool allUsersIntersect(OpResult result, Predicate predicate) {
  SmallVector<Value::user_range> stack = {result.getUsers()};
  if (stack.size() <= 1) return true;
  Operation *intersection = nullptr;
  llvm::SmallSet<Operation *, 32> visited;
  while (!stack.empty()) {
    auto range = stack.pop_back_val();
    llvm::SmallSet<Operation *, 8> unique;
    for (auto op : range) unique.insert(op);
    for (auto op : unique) {
      if (!visited.insert(op).second) {
        if (intersection == nullptr) intersection = op;
        else if (intersection != op) return false;
        continue;
      }
      if (op->hasTrait<OpTrait::ReturnLike>()) continue;
      if (!predicate(op)) return false;
      for (auto result : op->getResults()) stack.push_back(result.getUsers());
    }
  }
  return true;
}

// Find ParallelInsertSliceOp corresponding to the specified loop result.
inline tensor::ParallelInsertSliceOp
findParallelInsertSlice(scf::ForallOp &forall, OpResult loopRes) {
  BlockArgument outArg = forall.getTiedBlockArgument(loopRes);
  tensor::ParallelInsertSliceOp ins;
  for (auto &op : forall.getTerminator().getYieldingOps())
    if (auto i = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
        i && i.getDest() == outArg) {
      ins = i;
      break;
    }
  return ins;
}

} // namespace mlir::gc
#endif // GC_TRANSFORM_H
