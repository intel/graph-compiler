include_guard()

get_property(IMEX_INCLUDES GLOBAL PROPERTY IMEX_INCLUDES)
if (NOT DEFINED IMEX_INCLUDES)
    include(functions)
    set(IMEX_CHECK_LLVM_VERSION ON)
    set(IMEX_ENABLE_L0_RUNTIME 0)
    # TODO: Change to main https://github.com/oneapi-src/oneDNN.git when all the
    # required functionality is merged.
    gc_fetch_content(imex 496b240093b5e132b60c5ee69878300fe69be300 https://github.com/Menooker/mlir-extensions
            CMAKE_ARGS "-DMLIR_DIR=${MLIR_DIR};-DIMEX_CHECK_LLVM_VERSION=ON;-DIMEX_ENABLE_L0_RUNTIME=0"
    )

    set(IMEX_INCLUDES
            ${imex_BINARY_DIR}/include
            ${imex_SOURCE_DIR}/include
            ${imex_SOURCE_DIR}/src
    )
    set_property(GLOBAL PROPERTY IMEX_INCLUDES ${IMEX_INCLUDES})
endif ()
