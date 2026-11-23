//===-- Attributes.h - Graph Compiler IR attribute utilities --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_IR_ATTRIBUTES_H
#define GC_IR_ATTRIBUTES_H

#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>

namespace mlir::xegpu::uArch {
struct uArch;
}

namespace mlir::gc {

// ------------------- Attribute utilities ------------------ //
constexpr char GC_ATTR_KERNEL_NAME[] = "gc.kernel_name";
constexpr char GC_ATTR_LEVEL[] = "gc.tiling.level";
constexpr char GC_ATTR_WG_TILE_SIZES[] = "gc.tiling.wg_tile_sizes";
constexpr char GC_ATTR_NUM_KERNELS[] = "gc.num_kernels";
constexpr char GC_ATTR_LOOP_STAMP[] = "gc.loop.stamp";

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

  void erase(StringRef name) {
    if (!std::holds_alternative<NamedAttrList>(attrs)) {
      load();
      attrs = NamedAttrList(std::get<DictionaryAttr>(attrs));
    }
    std::get<NamedAttrList>(attrs).erase(name);
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

  const xegpu::uArch::uArch *getUarch();

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

  static const std::optional<const char *> getDeviceArch(int id);

private:
  static constexpr char ID[] = "id";
  static constexpr char NAME[] = "name";
  static constexpr char DEVICE_ARCH[] = "arch";
  static constexpr char MAX_WG_SIZE[] = "max_wg_size";
  static constexpr char SG_SIZES[] = "sg_sizes";
  const xegpu::uArch::uArch *arch = nullptr;
};

struct KernelCollectionAttrs : public GcAttrs<const char *> {
  static constexpr char PATH[] = "kernels";

  KernelCollectionAttrs(Operation *op) : GcAttrs<const char *>(op, PATH) {}
};

struct KernelAttrs : public GcAttrs<const char *, StringRef> {
  KernelAttrs(Operation *op, StringRef name)
      : GcAttrs<const char *, StringRef>(op, KernelCollectionAttrs::PATH,
                                         name) {}

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
} // namespace mlir::gc

#endif // GC_IR_ATTRIBUTES_H
