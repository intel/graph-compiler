add_mlir_library(GcGpuPasses
  LinalgToXeGPU.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

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
    GcUtilsIR
)

set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS GcGpuPasses)
