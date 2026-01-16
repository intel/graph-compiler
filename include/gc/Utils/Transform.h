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

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::gc {
constexpr char GC_ATTR_KERNEL_NAME[] = "gc.kernel_name";

template <typename T> auto createAttr(MLIRContext *ctx, T value) {
  if constexpr (std::is_integral_v<T>) {
    auto type = IntegerType::get(ctx, sizeof(T) * 8);
    return IntegerAttr::get(type, static_cast<int64_t>(value));
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

template <typename... Path> struct GcAttrs {
  GcAttrs(Operation *op, Path... p) : path(p...), op(op), attrs(nullptr) {}

  ~GcAttrs() { save(); }

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

  void save() {
    if (std::holds_alternative<NamedAttrList>(attrs)) {
      auto op = mod();
      std::apply(
          [&](auto... p) {
            NamedAttrList list(dyn_cast_if_present<DictionaryAttr>(
                op->getDiscardableAttr(ROOT)));
            save(op, list, std::get<NamedAttrList>(attrs), p...);
            if (list.empty())
              op->removeDiscardableAttr(ROOT);
            else
              op->setDiscardableAttr(ROOT, toDict(op, list));
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
    if (!isa<ModuleOp>(op))
      op = op->getParentOfType<ModuleOp>();
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
    if (newValues.empty())
      list.erase(name);
    else
      list.set(name, toDict(op, newValues));
  }

  template <typename T, typename... P>
  static void save(Operation *op, NamedAttrList &list,
                   const NamedAttrList &newValues, T name, P... path) {
    NamedAttrList newList(dyn_cast_if_present<DictionaryAttr>(list.get(name)));
    save(op, newList, newValues, path...);
    if (newList.empty())
      list.erase(name);
    else
      list.set(name, toDict(op, newList));
  }

  static inline DictionaryAttr toDict(Operation *op,
                                      const NamedAttrList &list) {
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
  void setArch(StringRef arch) { set(DEVICE_ARCH, arch); }

  std::optional<size_t> getVectorWidth() { return get<size_t>(VECTOR_WIDTH); }
  void setVectorWidth(size_t width) { set(VECTOR_WIDTH, width); }

  std::optional<size_t> getMaxWgSize() { return get<size_t>(MAX_WG_SIZE); }
  void setMaxWgSize(size_t size) { set(MAX_WG_SIZE, size); }

  std::optional<SmallVector<size_t>> getSgSizes() {
    return get<SmallVector<size_t>>(SG_SIZES);
  }
  void setSgSizes(ArrayRef<size_t> sizes) { set(SG_SIZES, sizes); }

  const std::optional<const char *> getDeviceArch(int deviceId) {
    // Using device ID from this source -
    // https://github.com/intel/compute-runtime/blob/master/shared/source/dll/devices/devices_base.inl
    switch (deviceId) {
    case 0x674C:
      return "cri";
    case 0xE202:
    case 0xE209:
    case 0xE20B:
    case 0xE20C:
    case 0xE20D:
    case 0xE210:
    case 0xE211:
    case 0xE212:
    case 0xE215:
    case 0xE216:
    case 0xE220:
    case 0xE221:
    case 0xE222:
    case 0xE223:
      return "bmg";
    case 0x0BD0:
    case 0x0BD5:
    case 0x0BD6:
    case 0x0BD7:
    case 0x0BD8:
    case 0x0BD9:
    case 0x0BDA:
    case 0x0BDB:
    case 0x0B69:
    case 0x0B6E:
    case 0x0BD4:
      return "pvc";
    default:
      return std::nullopt;
    }
  }

private:
  static constexpr char ID[] = "id";
  static constexpr char NAME[] = "name";
  static constexpr char DEVICE_ARCH[] = "arch";
  static constexpr char VECTOR_WIDTH[] = "vector_width";
  static constexpr char MAX_WG_SIZE[] = "max_wg_size";
  static constexpr char SG_SIZES[] = "sg_sizes";
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

  template <typename T = size_t> std::optional<T> getSgSize() {
    return get<T>(SG_SIZE);
  }

  template <typename T> void setSgSize(T sgSize) {
    set(SG_SIZE, static_cast<T>(sgSize));
  }

private:
  static constexpr char TILES[] = "tiles";
  static constexpr char THREADS[] = "threads";
  static constexpr char SG_SIZE[] = "sg_size";
};
} // namespace mlir::gc
#endif // GC_TRANSFORM_H