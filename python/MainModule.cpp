#include "gc-c/Dialects.h"
#include "gc-c/Passes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

PYBIND11_MODULE(_gc_mlir, m) {
  m.doc() = "Graph-compiler MLIR Python binding";

  mlirRegisterGraphCompilerPasses();
  //===----------------------------------------------------------------------===//
  // OneDNNGraph
  //===----------------------------------------------------------------------===//

  auto onednn_graphM = m.def_submodule("onednn_graph");
  onednn_graphM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__onednn_graph__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}