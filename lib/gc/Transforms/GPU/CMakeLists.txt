# todo: populate with default sources
# add_mlir_library(GCGPUPasses)

# todo: relax this to not require imex
if(GC_ENABLE_IMEX)
add_mlir_library(GCGPUIMEXPasses
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
    GCUtilsIR
)

set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS GCGPUIMEXPasses)
endif()
