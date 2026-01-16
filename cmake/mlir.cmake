include_guard()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${PROJECT_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${PROJECT_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${PROJECT_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

set(LLVM_TABLEGEN_FLAGS -I${PROJECT_BINARY_DIR}/include
                        -I${PROJECT_SOURCE_DIR}/include)

string(REPLACE " " ";" GC_LLVM_DEFINITIONS ${LLVM_DEFINITIONS})
target_compile_options(GcInterface INTERFACE ${GC_LLVM_DEFINITIONS})

# The dependency list copied from mlir/tools/mlir-runner/CMakeLists.txt of upstream
if(GC_DYLINK)
  set(LLVM_LINK_COMPONENTS LLVM)
  set(MLIR_LINK_COMPONENTS MLIR)
  set(MLIR_EXECUTION_ENGINE MLIRExecutionEngineShared)
else()
  set(LLVM_LINK_COMPONENTS
    Core
    Support
    nativecodegen
    native
  )
  set(MLIR_LINK_COMPONENTS
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRIR
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRToLLVMIRTranslationRegistration
    MLIRParser
    MLIRTargetLLVMIRExport
    MLIRSupport
    MLIRCAPIIR
    MLIRCAPITransforms
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
  )
  set(MLIR_EXECUTION_ENGINE MLIRExecutionEngine)
  get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
endif()
