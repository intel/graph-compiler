if(GC_DEV_LINK_DYN_LLVM)
  set(MLIR_LINK_COMPONENTS
    MLIR
  )
else()
  set(MLIR_LINK_COMPONENTS
    MLIRIR
  )
endif()

add_mlir_dialect_library(MLIROnednnGraph
    OnednnGraphDialect.cpp
    OnednnGraphOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc/Dialect/OnednnGraph
    
    DEPENDS
    MLIROnednnGraphOpsIncGen
    
    LINK_LIBS PUBLIC
    ${MLIR_LINK_COMPONENTS}
)
set_property(GLOBAL APPEND PROPERTY GC_DIALECT_LIBS MLIROnednnGraph)