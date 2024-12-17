include(imex)
get_property(IMEX_INCLUDES GLOBAL PROPERTY IMEX_INCLUDES)
get_property(IMEX_DIALECT_LIBS GLOBAL PROPERTY IMEX_DIALECT_LIBS)
get_property(IMEX_CONVERSION_LIBS GLOBAL PROPERTY IMEX_CONVERSION_LIBS)
set(IMEX_LIBS
  ${IMEX_CONVERSION_LIBS}
  IMEXUtil
  IMEXTransforms
  ${IMEX_DIALECT_LIBS}
)
set_property(GLOBAL APPEND PROPERTY IMEX_LIBS ${IMEX_LIBS})

gc_add_mlir_library(GcGpuPasses
  AddContextArg.cpp
  AllocsToSLM.cpp
  GpuTilingAndFusion.cpp
  GpuToGpuOcl.cpp
  LinalgToXeGPU.cpp
  Pipeline.cpp
  XeGPUDistribute.cpp

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    MLIRGPUDialect
    MLIRXeGPUDialect
    MLIRGPUTransforms
    MLIRGPUToSPIRV
    MLIRSCFToGPU
    MLIRSCFToSPIRV
    MLIRMathToSPIRV
    MLIRControlFlowToSPIRV
    MLIRMemRefTransforms
    GcInterface
    GcUtilsIR
    ${IMEX_LIBS}
)

foreach (include_dir IN LISTS IMEX_INCLUDES)
  list(APPEND IMEX_BUILD_INCLUDES $<BUILD_INTERFACE:${include_dir}>)
endforeach ()
target_include_directories(GcGpuPasses PUBLIC ${IMEX_BUILD_INCLUDES})
