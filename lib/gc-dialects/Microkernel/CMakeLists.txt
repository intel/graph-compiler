add_mlir_dialect_library(MLIRMicrokernel
    MicrokernelDialect.cpp
    MicrokernelOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc-dialects/Microkernel
    
    DEPENDS
    MLIRMicrokernelOpsIncGen
    
    LINK_LIBS PUBLIC
    MLIRIR
)