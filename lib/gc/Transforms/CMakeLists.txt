add_subdirectory(Utils)

gc_set_mlir_link_components(MLIR_LINK_COMPONENTS
  MLIRIR
  MLIRSupport
  MLIRBufferizationToMemRef
  MLIRBufferizationPipelines)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

gc_add_mlir_library(GcPasses
  MemRefToCPURuntime.cpp
  OneDNNGraphToLinalg.cpp
  Pipeline.cpp
  IterativeTilingAndFusion.cpp
  TilingUsingInterfaceX.cpp
  VerifyTargetDescription.cpp
  DeepTileContractionNamedOp.cpp
  TilingUtil.cpp
  SinkOpIntoInnerLoop.cpp
  MergeNestedForall.cpp
  StaticMemoryPlanning.cpp
  MergeAlloc.cpp
  MergeAllocTickBased.cpp

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    ${mlir_dialect_libs}
    ${mlir_conversion_libs}
    ${MLIR_LINK_COMPONENTS}
    ${GC_ONEDNN_DIALECT_LIB_NAME}
    GcInterface
  )

if(GC_ENABLE_IMEX)
  add_subdirectory(GPU)
endif()
