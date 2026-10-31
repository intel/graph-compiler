gc_add_mlir_library(
  GcGpuOclSourceRuntime
  OclSourceModule.cpp
  LINK_LIBS
  PUBLIC
  GcInterface
  GcGpuCommonPasses
  OpenCL::OpenCL)
