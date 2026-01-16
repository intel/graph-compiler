gc_add_mlir_library(GcGpuOclPasses
  GpuToGpuOcl.cpp

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    ${MLIR_LINK_COMPONENTS}
    ${MLIR_DIALECT_LIBS}
    ${MLIR_CONVERSION_LIBS}
    GcInterface
)
