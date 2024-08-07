gc_set_mlir_link_components(MLIR_LINK_COMPONENTS
  MLIRIR
  MLIRSupport)

gc_add_mlir_library(GcAnalysis
  TargetDescriptionAnalysis.cpp
  DataFlow/ConstantSubgraphAnalyser.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    ${mlir_dialect_libs}
    ${MLIR_LINK_COMPONENTS}
    GcInterface
  )
