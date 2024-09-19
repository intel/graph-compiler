gc_add_mlir_library(GcGpuPasses
  AddContextArg.cpp
  GpuToGpuOcl.cpp
  LinalgToXeGPU.cpp
  Pipeline.cpp

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
)

include(imex)
get_property(IMEX_INCLUDES GLOBAL PROPERTY IMEX_INCLUDES)
target_include_directories(GcGpuPasses PRIVATE ${IMEX_INCLUDES})

