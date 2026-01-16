#include <nanobind/nanobind.h>

#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include "gc/Utils/Error.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::gc::gpu;

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

struct GpuContext {
  const OclRuntime runtime;
  OclContext oclCtx;
  MLIRContext mlirCtx{gc::getDialectRegistry()};

  static GpuContext &get() {
    static GpuContext instance{gcGetOrReport(OclRuntime::get())};
    return instance;
  }

private:
  explicit GpuContext(OclRuntime rt)
      : runtime(std::move(rt)),
        oclCtx{runtime, gcGetOrReport(runtime.createQueue())} {}
};

struct Usm {
  mutable void *ptr;
  mutable size_t size;

  Usm(void *ptr = nullptr, size_t size = 0) : ptr(ptr), size(size) {}

  Usm(size_t size, bool shared = false)
      : ptr(gcGetOrReport(shared
                              ? GpuContext::get().runtime.usmAllocShared(size)
                              : GpuContext::get().runtime.usmAllocDev(size))),
        size(size) {}

  explicit Usm(const nb::object &tensor, bool shared = false)
      : Usm(tensorSize(tensor), shared) {
    nb::object cpu = tensor.attr("cpu")().attr("contiguous")();
    auto addr = nb::cast<uintptr_t>(cpu.attr("data_ptr")());
    copyFrom(reinterpret_cast<void *>(addr), size);
  }

  // Non-copyable
  Usm(const Usm &) = delete;
  Usm &operator=(const Usm &) = delete;
  // Movable
  Usm(Usm &&other) noexcept : ptr(other.ptr), size(other.size) {
    other.ptr = nullptr;
  }
  Usm &operator=(Usm &&other) noexcept {
    if (this != &other) {
      if (ptr) {
        gcGetOrReport(GpuContext::get().runtime.usmFree(ptr));
      }
      ptr = other.ptr;
      size = other.size;
      other.ptr = nullptr;
      other.size = 0;
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
    auto &ctx = GpuContext::get();
    gcGetOrReport(ctx.runtime.usmCpy(ctx.oclCtx, src, ptr, size));
    gcGetOrReport(GpuContext::get().oclCtx.finish());
  }

  void copyTo(void *dst, size_t size) const {
    auto &ctx = GpuContext::get();
    gcGetOrReport(ctx.runtime.usmCpy(ctx.oclCtx, ptr, dst, size));
    gcGetOrReport(GpuContext::get().oclCtx.finish());
  }

  ~Usm() {
    if (ptr) {
      gcGetOrReport(GpuContext::get().runtime.usmFree(ptr));
    }
  }
};

using GpuModule = std::shared_ptr<const OclModule>;
NB_MODULE(graph_compiler, m) {
  m.doc() = "Graph Compiler";

  m.def(
      "ualloc",
      [](size_t size, bool shared = false) { return Usm(size, shared); },
      nb::arg("size"), nb::arg("shared") = false,
      "Allocate USM memory of the given size.");

  nb::class_<GpuModule>(m, "GpuModule")
      .def(nb::new_([](nb::str mlir, bool dump = false, bool wait = false) {
             auto &ctx = GpuContext::get();
             auto mlirMod = mlir::parseSourceString<ModuleOp>(
                 std::string(mlir.c_str()), &ctx.mlirCtx);
             if (!mlirMod) {
               throw std::runtime_error("Failed to parse MLIR module");
             }

             OclModuleBuilderOpts builderOpts;
             builderOpts.dumpIr = dump;
             builderOpts.pipeline = [wait](OpPassManager &pm,
                                           gc::GPUPipelineOptions &opts) {
               opts.isUsmArgs = true;
               opts.callFinish = wait;
               populateGPUPipeline(pm, opts);
             };
             OclModuleBuilder builder{mlirMod, builderOpts};
             auto oclMod = gcGetOrReport(builder.build(ctx.runtime));
             assert(oclMod->isStatic);
             return oclMod;
           }),
           nb::arg("mod"), nb::arg("dump") = false, nb::arg("wait") = false)

      .def("__call__", [](const GpuModule &mod, nb::args args) {
        SmallVector<Usm> usms;
        SmallVector<std::tuple<Usm &, nb::object, nb::object>> outputs;
        StaticExecutor<8> exec{mod};

        for (size_t i = 0; i < args.size(); ++i) {
          nb::object arg = args[i];
          void *ptr;
          if (isTensor(arg)) {
            nb::object cpu = arg.attr("cpu")().attr("contiguous")();
            usms.emplace_back(cpu);
            ptr = usms.back().ptr;
            if (mod->isOutputArg(i)) {
              outputs.emplace_back(usms.back(), std::move(arg), std::move(cpu));
            }
          } else {
            ptr = nb::cast<Usm &>(arg).ptr;
          }
          exec.arg(ptr, true);
        }

        exec(GpuContext::get().oclCtx);

        // Copy back to tensors
        if (!outputs.empty()) {
          for (auto &[usm, tensor, cpu] : outputs) {
            usm.to(cpu);
            tensor.attr("copy_")(cpu);
          }
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