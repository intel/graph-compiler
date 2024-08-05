gc_add_mlir_library(GcGpuPasses
  GPULegalizeModule.cpp
  ConvertGpuSignaturesToLLVM.cpp
  GPUAttachGenTarget.cpp

  PARTIAL_SOURCES_INTENDED
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
    GcInterface
    GcUtilsIR
)

if(GC_ENABLE_IMEX)
gc_add_mlir_library(GcIMEXPasses
  LinalgToXeGPU.cpp

  PARTIAL_SOURCES_INTENDED
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
endif()
