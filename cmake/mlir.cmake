include_guard()

find_package(MLIR REQUIRED CONFIG)

if(NOT GC_DYLINK)
  set(LLVM_LINK_LLVM_DYLIB OFF)
  set(MLIR_LINK_MLIR_DYLIB OFF)
endif()

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

# GCC's uninitialized use analysis produces false positives on LLVM's ADT.
# Upstream disables it in llvm/cmake/config-ix.cmake, which is not exported for
# out-of-tree builds, so setting the variable for 'HandleLLVMOptions' manually
# (flang does the same for its standalone build).
if(CMAKE_COMPILER_IS_GNUCXX)
  set(USE_NO_MAYBE_UNINITIALIZED 1)
endif()

include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
foreach(dir IN LISTS LLVM_INCLUDE_DIRS MLIR_INCLUDE_DIRS)
  target_include_directories(GcInterface SYSTEM
                             INTERFACE $<BUILD_INTERFACE:${dir}>)
endforeach()

set(LLVM_TABLEGEN_FLAGS -I${PROJECT_BINARY_DIR}/include
                        -I${PROJECT_SOURCE_DIR}/include)

string(REPLACE " " ";" GC_LLVM_DEFINITIONS ${LLVM_DEFINITIONS})
target_compile_options(GcInterface INTERFACE ${GC_LLVM_DEFINITIONS})

function(gc_remove_dylib_aggregate_deps)
  set(worklist ${ARGN})
  set(seen)
  while(worklist)
    list(POP_FRONT worklist target)
    if(NOT TARGET "${target}" OR target IN_LIST seen)
      continue()
    endif()
    list(APPEND seen "${target}")

    get_target_property(libs "${target}" INTERFACE_LINK_LIBRARIES)
    if(libs)
      set(filtered_libs)
      foreach(lib IN LISTS libs)
        if(lib STREQUAL "LLVM"
           OR lib STREQUAL "MLIR"
           OR lib STREQUAL "$<LINK_ONLY:LLVM>"
           OR lib STREQUAL "$<LINK_ONLY:MLIR>"
           OR lib MATCHES "lib(LLVM|MLIR)\\.so")
          continue()
        endif()
        list(APPEND filtered_libs "${lib}")
        if(TARGET "${lib}")
          list(APPEND worklist "${lib}")
        endif()
      endforeach()
      set_target_properties("${target}" PROPERTIES INTERFACE_LINK_LIBRARIES
                                                   "${filtered_libs}")
    endif()

    get_target_property(configs "${target}" IMPORTED_CONFIGURATIONS)
    foreach(config IN LISTS configs)
      get_target_property(deps "${target}"
                          IMPORTED_LINK_DEPENDENT_LIBRARIES_${config})
      if(deps)
        list(REMOVE_ITEM deps LLVM MLIR)
        list(FILTER deps EXCLUDE REGEX "lib(LLVM|MLIR)\\.so")
        set_target_properties(
          "${target}" PROPERTIES IMPORTED_LINK_DEPENDENT_LIBRARIES_${config}
                                 "${deps}")
      endif()
    endforeach()
  endwhile()
endfunction()

if(GC_DYLINK)
  set(GC_LLVM_LINK_COMPONENTS LLVM)
  set(MLIR_LINK_COMPONENTS MLIR)
  set(MLIR_EXECUTION_ENGINE MLIRExecutionEngineShared)
else()
  set(GC_LLVM_LINK_COMPONENTS Core Support nativecodegen native)
  set(MLIR_LINK_COMPONENTS MLIRAnalysis MLIRIR MLIRLLVMDialect MLIRParser
                           MLIRTargetLLVMIRExport MLIRSupport)
  set(MLIR_EXECUTION_ENGINE MLIRExecutionEngine)
  set(MLIR_DIALECT_LIBS
      MLIRAffineDialect
      MLIRAffineTransforms
      MLIRArithDialect
      MLIRArithTransforms
      MLIRArithValueBoundsOpInterfaceImpl
      MLIRBufferizationAllExtensions
      MLIRBufferizationDialect
      MLIRBufferizationTransforms
      MLIRControlFlowDialect
      MLIRControlFlowTransforms
      MLIRDLTIDialect
      MLIRFuncAllExtensions
      MLIRFuncDialect
      MLIRGPUDialect
      MLIRGPUPipelines
      MLIRGPUTransforms
      MLIRIndexDialect
      MLIRLinalgDialect
      MLIRLinalgTransforms
      MLIRMathDialect
      MLIRMemRefDialect
      MLIRMemRefTransforms
      MLIRSCFDialect
      MLIRSCFTransforms
      MLIRShapeDialect
      MLIRTensorAllExtensions
      MLIRTensorDialect
      MLIRTensorInferTypeOpInterfaceImpl
      MLIRTensorTilingInterfaceImpl
      MLIRUBDialect
      MLIRVectorDialect
      MLIRVectorTransforms
      MLIRXeGPUDialect
      MLIRXeGPUTransforms
      MLIRXeGPUUtils
      MLIRXeVMDialect
      MLIRXeVMTarget)
  set(MLIR_CONVERSION_LIBS
      MLIRArithToLLVM
      MLIRBuiltinToLLVMIRTranslation
      MLIRComplexToLLVM
      MLIRControlFlowToLLVM
      MLIRFuncToLLVM
      MLIRGPUToGPURuntimeTransforms
      MLIRGPUToLLVMIRTranslation
      MLIRIndexToLLVM
      MLIRLLVMToLLVMIRTranslation
      MLIRMathToLLVM
      MLIRMemRefToLLVM
      MLIRSCFToGPU
      MLIRUBToLLVM
      MLIRVectorToLLVM
      MLIRVectorToLLVMPass
      MLIRVectorToXeGPU
      MLIRXeVMToLLVMIRTranslation)
endif()

if(GC_DYLINK)
  target_link_libraries(GcInterface INTERFACE LLVM MLIR
                                              ${MLIR_EXECUTION_ENGINE})
else()
  gc_remove_dylib_aggregate_deps(
    ${MLIR_LINK_COMPONENTS} ${MLIR_DIALECT_LIBS} ${MLIR_CONVERSION_LIBS}
    ${MLIR_EXECUTION_ENGINE})
  llvm_map_components_to_libnames(GC_LLVM_LIBS ${GC_LLVM_LINK_COMPONENTS})
  target_link_libraries(
    GcInterface
    INTERFACE ${MLIR_LINK_COMPONENTS}
              ${MLIR_DIALECT_LIBS}
              ${MLIR_CONVERSION_LIBS}
              ${MLIR_EXECUTION_ENGINE}
              ${GC_LLVM_LIBS}
              LLVMX86CodeGen
              LLVMX86AsmParser
              LLVMSPIRVCodeGen)
endif()
