gc_set_mlir_link_components(MLIR_LINK_COMPONENTS MLIRIR)

add_mlir_dialect_library(MLIRLinalgxTransforms
    AllInterfaces.cpp
    BufferizableOpInterfaceImpl.cpp
    TilingInterfaceImpl.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc/Dialect/Linalgx/Transforms
    
    DEPENDS
    MLIRLinalgx
    MLIRLinalgDialect
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
  
    LINK_LIBS PUBLIC
    ${MLIR_LINK_COMPONENTS}
)
set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS MLIRLinalgxTransforms)