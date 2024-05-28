gc_set_mlir_link_components(MLIR_LINK_COMPONENTS MLIRIR)

include(onednn)

message(WARNING ${dnnl_SOURCE_DIR})

include_directories(
        ${dnnl_SOURCE_DIR}/include
        ${dnnl_BINARY_DIR}/include
)

add_mlir_dialect_library(MLIRMicrokernelTransforms
	ConvertLinalgToMicrokernel.cpp
        ConvertMicrokernelToDnnlFunc.cpp
        EarlyDispatchMicrokernel.cpp
        MicrokernelInvariantCodeMotion.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/

        DEPENDS
	MLIRMicrokernelPassesIncGen

        LINK_LIBS PUBLIC
        ${MLIR_LINK_COMPONENTS}
	GCMLIRUtils
        )
