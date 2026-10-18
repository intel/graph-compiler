//===-- Nanobind.cpp - Python bindings ------------------------*- C++ -*-===--//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

#include <llvm/ADT/SmallVector.h>
#include <nanobind/nanobind.h>

#include "gc/GpuCompiler.h"
#include "gc/Runtime/ZeGpuRuntime.h"

namespace nb = nanobind;
using namespace gc::gpu;

static bool isTensor(const nb::object &obj) {
  static nb::object torchTensor = []() -> nb::object {
    try {
      return nb::module_::import_("torch").attr("Tensor");
    } catch (...) {
      return nb::none();
    }
  }();
  return torchTensor && nb::isinstance(obj, torchTensor);
}

static size_t tensorSize(const nb::object &tensor) {
  if (!isTensor(tensor)) {
    throw std::invalid_argument("Not a tensor!");
  }
  return nb::cast<size_t>(tensor.attr("element_size")()) *
         nb::cast<size_t>(tensor.attr("nelement")());
}

static ZeGpuRuntime *getRuntime() {
  if (auto runtime = ZeGpuRuntime::get()) return runtime;
  throw std::runtime_error("No suitable GPU device found");
}

struct Usm {
  void *ptr{nullptr};
  size_t size{0};
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> strides;

  Usm() = default;

  Usm(size_t size, bool shared = false) : size(size) {
    auto runtime = getRuntime();
    ptr =
        shared ? runtime->allocShared(size, 0) : runtime->allocDevice(size, 0);
    if (!ptr && size) throw std::bad_alloc();
  }

  explicit Usm(const nb::object &tensor, bool shared = false)
      : Usm(tensorSize(tensor), shared) {
    nb::object cpu = tensor.attr("cpu")().attr("contiguous")();
    setLayout(cpu);
    auto addr = nb::cast<uintptr_t>(cpu.attr("data_ptr")());
    copyFrom(reinterpret_cast<void *>(addr), size);
  }

  // Non-copyable
  Usm(const Usm &) = delete;
  Usm &operator=(const Usm &) = delete;
  // Movable
  Usm(Usm &&other) noexcept
      : ptr(std::exchange(other.ptr, nullptr)),
        size(std::exchange(other.size, 0)), shape(std::move(other.shape)),
        strides(std::move(other.strides)) {}
  Usm &operator=(Usm &&other) noexcept {
    if (this != &other) {
      release();
      ptr = std::exchange(other.ptr, nullptr);
      size = std::exchange(other.size, 0);
      shape = std::move(other.shape);
      strides = std::move(other.strides);
    }
    return *this;
  }

  void to(const nb::object &tensor) const {
    auto tensorSize = ::tensorSize(tensor);
    if (tensorSize != size) {
      throw std::invalid_argument("Tensor size does not match USM size!");
    }
    auto cpu = tensor.attr("cpu")().attr("contiguous")();
    size_t cpuAddr = nb::cast<size_t>(cpu.attr("data_ptr")());
    copyTo(reinterpret_cast<void *>(cpuAddr), size);
    tensor.attr("copy_")(cpu);
  }

  void copyFrom(const void *src, size_t size) const {
    auto runtime = getRuntime();
    auto *event = runtime->memcpy(src, ptr, size, nullptr, 0);
    if (event) runtime->wait(&event, 1);
  }

  void copyTo(void *dst, size_t size) const {
    auto runtime = getRuntime();
    auto *event = runtime->memcpy(ptr, dst, size, nullptr, 0);
    if (event) runtime->wait(&event, 1);
  }

  unsigned rank() const { return shape.size(); }

  ~Usm() { release(); }

private:
  void setLayout(const nb::object &tensor) {
    auto rank = nb::cast<unsigned>(tensor.attr("dim")());
    shape.reserve(rank);
    strides.reserve(rank);
    auto shapeSize = tensor.attr("size");
    auto strideSize = tensor.attr("stride");
    for (unsigned i = 0; i < rank; ++i) {
      shape.push_back(nb::cast<int64_t>(shapeSize(i)));
      strides.push_back(nb::cast<int64_t>(strideSize(i)));
    }
  }

  void release() {
    if (ptr) getRuntime()->free(ptr);
    ptr = nullptr;
    size = 0;
  }
};

struct GpuModule {
  GpuModule(FutureProgram &&prog) : prog(std::move(prog)) {}

  Program *program() {
    if (std::holds_alternative<FutureProgram>(prog)) {
      std::shared_ptr<Program> p = std::get<FutureProgram>(prog).get();
      prog = p;
      return p.get();
    }
    return std::get<std::shared_ptr<Program>>(prog).get();
  }

private:
  std::variant<FutureProgram, std::shared_ptr<Program>> prog;
};

NB_MODULE(graph_compiler, m) {
  m.doc() = "Graph Compiler";

  m.def(
      "ualloc",
      [](size_t size, bool shared = false) { return Usm(size, shared); },
      nb::arg("size"), nb::arg("shared") = false,
      "Allocate USM memory of the given size.");

  nb::class_<GpuModule>(m, "GpuModule")
      .def(nb::new_([](nb::str mlir, bool dump = false) {
             (void)dump;
             GpuCompiler::Options opts;
             opts.wait = true;
             opts.dump = dump;
             return GpuModule{
                 GpuCompiler::get().compile<ZeGpuRuntime>(mlir.c_str(), opts)};
           }),
           nb::arg("mod"), nb::arg("dump") = false)

      .def("__call__", [](GpuModule &mod, nb::args args) {
        llvm::SmallVector<std::unique_ptr<Usm>> tensorUsms;
        llvm::SmallVector<nb::object> tensors;
        llvm::SmallVector<nb::object> cpus;
        SmallBuilder<> builder;

        for (size_t i = 0; i < args.size(); ++i) {
          nb::object arg = args[i];
          Usm *usm;
          if (isTensor(arg)) {
            nb::object cpu = arg.attr("cpu")().attr("contiguous")();
            tensorUsms.emplace_back(std::make_unique<Usm>(cpu));
            usm = tensorUsms.back().get();
            tensors.push_back(std::move(arg));
            cpus.push_back(std::move(cpu));
          } else {
            usm = &nb::cast<Usm &>(arg);
          }
          builder.storeMemref(usm->ptr, usm->rank(), usm->shape.data(),
                              usm->strides.data());
        }

        {
          nb::gil_scoped_release release;
          auto program = mod.program();
          auto runtime = getRuntime();
          program->main(builder.build(*program), runtime);
        }

        for (size_t i = 0; i < tensors.size(); ++i) {
          tensorUsms[i]->to(cpus[i]);
          tensors[i].attr("copy_")(cpus[i]);
        }
      });

  nb::class_<Usm>(m, "Usm")
      .def(nb::init<size_t, bool>(), nb::arg("size"), nb::arg("shared") = false,
           "Allocate USM memory of the given size.")
      .def(nb::init<nb::object, bool>(), nb::arg("tensor"),
           nb::arg("shared") = false,
           "Allocate USM memory and copy the data from tensor.")
      .def("to", &Usm::to, nb::arg("tensor"),
           "Copy the data from USM to the given tensor.");
}
