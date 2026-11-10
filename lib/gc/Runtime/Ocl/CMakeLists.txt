gc_add_mlir_library(
  GcOclRuntime
  OclRuntime.cpp
  LINK_LIBS
  PUBLIC
  GcInterface
  OpenCL::OpenCL)
