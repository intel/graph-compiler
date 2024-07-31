add_mlir_library(GCGPUPasses
  GPULegalizeModule.cpp
  ConvertGpuSignaturesToLLVM.cpp
  GPUAttachGenTarget.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    MLIRGENDialect
    MLIRGENTarget
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

set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS GCGPUPasses)
