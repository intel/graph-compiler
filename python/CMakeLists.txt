include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `gc_mlir`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=gc_mlir.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(GcPythonSources)


################################################################################
# Dialect bindings
################################################################################

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT GcPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gc_mlir"
    TD_FILE dialects/OneDNNGraphOps.td
    SOURCES 
      dialects/onednn_graph.py
    DIALECT_NAME onednn_graph
    )

declare_mlir_python_extension(GcPythonSources.Extension
  MODULE_NAME _gc_mlir
  ADD_TO_PARENT GcPythonSources
  SOURCES
    MainModule.cpp
  EMBED_CAPI_LINK_LIBS
    GcCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(GcPythonCAPI
  INSTALL_COMPONENT GcPythonModules
  INSTALL_DESTINATION python_packages/gc_mlir_core/gc_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/gc_mlir_core/gc_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    GcPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(GcPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/gc_mlir_core/gc_mlir"
  INSTALL_PREFIX "python_packages/gc_mlir_core/gc_mlir"
  DECLARED_SOURCES
    GcPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    GcPythonCAPI
  )