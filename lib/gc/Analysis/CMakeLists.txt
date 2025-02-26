gc_set_mlir_link_components(MLIR_LINK_COMPONENTS
  MLIRIR
  MLIRSupport)

gc_add_mlir_library(GcAnalysis
  TargetDescriptionAnalysis.cpp
  MatmulConfigAnalysis.cpp
  GlobalAnalysis.cpp

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    ${mlir_dialect_libs}
    ${MLIR_LINK_COMPONENTS}
    GcInterface
)
