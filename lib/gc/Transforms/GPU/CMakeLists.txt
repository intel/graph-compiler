add_subdirectory(OCL)

get_property(GC_PASS_LIBS GLOBAL PROPERTY GC_PASS_LIBS)
get_property(GC_DIALECT_LIBS GLOBAL PROPERTY GC_DIALECT_LIBS)
gc_add_mlir_library(
  GcGpuPasses
  tiling/TileContraction.cpp
  tiling/TileAttention.cpp
  tiling/TileParallel.cpp
  AddContextArg.cpp
  Decomposition.cpp
  GpuDeviceProps.cpp
  GpuKernelOutline.cpp
  Vectorize.cpp
  RemoveAllocs.cpp
  HoistForLoopTransferRead.cpp
  AttentionOptimizations/Utils.cpp
  AttentionOptimizations/SetAttentionLayouts.cpp
  AttentionOptimizations/SetAttentionPrefetch.cpp
  AttentionOptimizations/HoistAttentionVLoad.cpp
  SetGpuFastMath.cpp
  Pipeline.cpp
  DEPENDS
  GraphCompilerPassIncGen
  LINK_LIBS
  PUBLIC
  ${MLIR_LINK_COMPONENTS}
  ${MLIR_DIALECT_LIBS}
  ${MLIR_CONVERSION_LIBS}
  ${GC_PASS_LIBS}
  ${GC_DIALECT_LIBS}
  GcInterface
  GcGpuOclPasses)
