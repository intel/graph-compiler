gc_set_mlir_link_components(MLIR_LINK_COMPONENTS MLIRIR)

include(onednn)

get_property(DNNL_INCLUDES GLOBAL PROPERTY DNNL_INCLUDES)

include_directories(${DNNL_INCLUDES})

add_mlir_dialect_library(MLIRMicrokernelTransforms
	ConvertLinalgToMicrokernel.cpp
        ConvertMicrokernelToDnnlFunc.cpp
        EarlyDispatchMicrokernel.cpp
        MicrokernelInvariantCodeMotion.cpp
        MergeBranchMicrokernelContext.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/

        DEPENDS
	MLIRMicrokernelPassesIncGen

        LINK_LIBS PUBLIC
        ${MLIR_LINK_COMPONENTS}
	GCUtilsIR
        )

set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS MLIRMicrokernelTransforms)
