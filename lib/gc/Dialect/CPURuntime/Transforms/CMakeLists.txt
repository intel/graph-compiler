gc_set_mlir_link_components(MLIR_LINK_COMPONENTS MLIRFuncDialect)

add_mlir_dialect_library(MLIRCPURuntimeTransforms
        CPURuntimeToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/

        DEPENDS
        MLIRCPURuntimePassesIncGen

        LINK_LIBS PUBLIC
        ${MLIR_LINK_COMPONENTS}
        MLIRCPURuntimeDialect
        )

set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS MLIRCPURuntimeTransforms)