gc_set_mlir_link_components(MLIR_LINK_COMPONENTS
  MLIRIR
  MLIRSupport
  MLIRBufferizationToMemRef
  MLIRBufferizationPipelines)

add_mlir_library(GCPasses
  OneDNNGraphToLinalg.cpp
  Pipeline.cpp
  TileNamed.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    ${mlir_dialect_libs}
    ${MLIR_LINK_COMPONENTS}
    MLIROneDNNGraph
  )

set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS GCPasses)
