set(IMEX_LIBS "")
if(GC_ENABLE_IMEX)
add_subdirectory(IMEX)
set(IMEX_LIBS GcGpuIMEXPasses)
endif()

set(OCL_PASSES_LIBS "")
if(GC_ENABLE_GPU)
add_subdirectory(OCL)
set(OCL_PASSES_LIBS GcGpuOCLPasses)
endif()

gc_add_mlir_library(GcGpuPasses
  AddContextArg.cpp
  AllocsToSLM.cpp
  GpuTilingAndFusion.cpp
  Pipeline.cpp
  XeVMAttachTarget.cpp

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
    MLIRXeVMToLLVMIRTranslation
    GcInterface
    GcUtilsIR
    ${IMEX_LIBS}
    ${OCL_PASSES_LIBS}
)
