add_mlir_dialect_library(MLIRCPURuntimeDialect
        CPURuntimeDialect.cpp
        CPURuntimeOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/

        DEPENDS
        MLIRCPURuntimeOpsIncGen
        MLIRCPURuntimePassesIncGen

        LINK_LIBS PUBLIC
        MLIR
        # MLIRInferTypeOpInterface
        # MLIRFuncDialect
        )
