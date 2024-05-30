add_subdirectory(Utils)

gc_set_mlir_link_components(MLIR_LINK_COMPONENTS
  MLIRIR
  MLIRSupport
  MLIRBufferizationToMemRef
  MLIRBufferizationPipelines)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

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
    ${mlir_conversion_libs}
    ${MLIR_LINK_COMPONENTS}
    MLIROneDNNGraph
  )

set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS GCPasses)

add_subdirectory(GPU)
add_subdirectory(Microkernel)
