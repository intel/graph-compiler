gc_add_mlir_library(
  GcGpuCommonPasses
  PARTIAL_SOURCES_INTENDED
  Pipeline.cpp
  SPIRVPipeline.cpp
  GpuToGcGpu.cpp
  MemrefCopyToGpuMemcpy.cpp
  SourcePipeline.cpp
  Decomposition.cpp
  TensorConcatToLinalg.cpp
  DEPENDS
  GraphCompilerPassIncGen
  MLIRLinalgxOpsIncGen
  LINK_LIBS
  PUBLIC
  GcInterface)

if(GC_ENABLE_OCL)
  target_sources(obj.GcGpuCommonPasses PRIVATE OclSource.cpp)
endif()

if(NOT GC_ENABLE_IGC)
  return()
endif()

add_dependencies(obj.GcGpuCommonPasses GcTensorMaskingOpInterfaceIncGen)

get_property(GC_PASS_LIBS GLOBAL PROPERTY GC_PASS_LIBS)
get_property(GC_DIALECT_LIBS GLOBAL PROPERTY GC_DIALECT_LIBS)
gc_add_mlir_library(
  GcGpuPasses
  PARTIAL_SOURCES_INTENDED
  tiling/TileContraction.cpp
  tiling/TileAttention.cpp
  tiling/TileGroupedMatmul.cpp
  tiling/TileParallel.cpp
  GuardEmptyTiles.cpp
  ApplyPaddingLevel.cpp
  TensorMaskingOpInterface.cpp
  DropUnitDims.cpp
  GpuDeviceProps.cpp
  GpuKernelOutline.cpp
  TensorReshapeToLinalg.cpp
  Vectorize.cpp
  RemoveAllocs.cpp
  SetKernelDepends.cpp
  HoistForLoopTransferRead.cpp
  FoldPadRoundtrip.cpp
  SimplifyIndexingMaps.cpp
  HoistStaticForInit.cpp
  PeeledForToIf.cpp
  NormalizeMemrefLayouts.cpp
  AttentionOptimizations/Utils.cpp
  AttentionOptimizations/SetAttentionLayouts.cpp
  AttentionOptimizations/SetAttentionPrefetch.cpp
  AttentionOptimizations/HoistAttentionVLoad.cpp
  SetLayouts.cpp
  SetGpuFastMath.cpp
  DEPENDS
  GraphCompilerPassIncGen
  GcTensorMaskingOpInterfaceIncGen
  LINK_LIBS
  PUBLIC
  GcGpuCommonPasses
  GcInterface)
