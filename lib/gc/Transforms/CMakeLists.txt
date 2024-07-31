add_subdirectory(Utils)

gc_set_mlir_link_components(MLIR_LINK_COMPONENTS
  MLIRIR
  MLIRSupport
  MLIRBufferizationToMemRef
  MLIRBufferizationPipelines)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

gc_add_mlir_library(GcPasses
  OneDNNGraphToLinalg.cpp
  Pipeline.cpp
  TileNamed.cpp

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    ${mlir_dialect_libs}
    ${mlir_conversion_libs}
    ${MLIR_LINK_COMPONENTS}
    ${GC_ONEDNN_DIALECT_LIB_NAME}
    GcInterface
  )

if(GC_ENABLE_GPU)
  add_subdirectory(GPU)
endif()
