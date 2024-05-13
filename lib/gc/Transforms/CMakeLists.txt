if(GC_LINK_DYNAMIC_LLVM)
  set(MLIR_LINK_COMPONENTS
    MLIR
  )
else()
  set(MLIR_LINK_COMPONENTS
    MLIRIR
    MLIRSupport
    MLIRBufferizationToMemRef
    MLIRBufferizationPipelines
  )
endif()

add_mlir_library(GCPasses
  TileNamed.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    ${mlir_dialect_libs}
    ${MLIR_LINK_COMPONENTS}
  )
set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS GCPasses)
