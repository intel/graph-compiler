gc_set_mlir_link_components(MLIR_LINK_COMPONENTS MLIRIR)

add_mlir_dialect_library(MLIRMicrokernel
    MicrokernelEnum.cpp
    MicrokernelDialect.cpp
    MicrokernelOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc/Dialect/Microkernel
    
    DEPENDS
    MLIRMicrokernelOpsIncGen
    MLIRMicrokernelPassesIncGen
    
    LINK_LIBS PUBLIC
    ${MLIR_LINK_COMPONENTS}
    GCUtilsIR
)
set_property(GLOBAL APPEND PROPERTY GC_DIALECT_LIBS MLIRMicrokernel)
