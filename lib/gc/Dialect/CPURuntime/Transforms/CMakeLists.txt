add_mlir_dialect_library(MLIRCPURuntimeTransforms
        CPURuntimePasses.cpp
        CPURuntimeToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/

        DEPENDS
        MLIRCPURuntimePassesIncGen

        LINK_LIBS PUBLIC
        MLIRFuncDialect
        MLIRCPURuntimeDialect
        )

set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS MLIRCPURuntimeTransforms)