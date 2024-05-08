add_mlir_library(GCPasses
  TileNamed.cpp
  MergeAlloc.cpp
  StaticMemoryPlanning.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc-dialects

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    ${mlir_dialect_libs}
    MLIRIR
    MLIRSupport
    MLIRBufferizationToMemRef
    MLIRBufferizationPipelines
  )
