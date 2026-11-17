gc_add_mlir_library(
  GcIR
  GCAttributes.cpp
  DEPENDS
  GcIRIncGen
  LINK_LIBS
  PUBLIC
  GcInterface)
