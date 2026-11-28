gc_add_mlir_library(
  GcUtils
  Cache.cpp
  LINK_COMPONENTS
  Support
  LINK_LIBS
  PUBLIC
  GcInterface)
