include_guard()

get_property(DNNL_INCLUDES GLOBAL PROPERTY DNNL_INCLUDES)
if (NOT DEFINED DNNL_INCLUDES)
    include(functions)

    # TODO: Change to main https://github.com/oneapi-src/oneDNN.git when all the
    # required functionality is merged.
    gc_fetch_content(dnnl dev https://github.com/kurapov-peter/oneDNN.git
            SKIP_ADD
            CMAKE_ARGS -DDNNL_IS_MAIN_PROJECT=FALSE -DDNNL_BUILD_TESTS=FALSE -DDNNL_BUILD_EXAMPLES=FALSE
    )

    set(DNNL_INCLUDES
            ${dnnl_BINARY_DIR}/include
            ${dnnl_SOURCE_DIR}/include
            ${dnnl_SOURCE_DIR}/src
    )
    set_property(GLOBAL PROPERTY DNNL_INCLUDES ${DNNL_INCLUDES})
    set_property(GLOBAL PROPERTY DNNL_SOURCE_DIR ${dnnl_SOURCE_DIR})

    # This allows to generate headers from *.in without adding the library to the build.
    # If the build is required, remove this and the SKIP_ADD option above.
    # if (DEFINED CMAKE_GENERATOR)
    #    set(GENERATOR_FLAG "-G ${CMAKE_GENERATOR}")
    # endif ()
    # execute_process(COMMAND ${CMAKE_COMMAND} ${GENERATOR_FLAG}
    #        -Wno-dev
    #        -S ${dnnl_SOURCE_DIR}
    #        -B ${dnnl_BINARY_DIR}
    #        ${GC_DNNL_CMAKE_ARGS}
    # )
    
    include(onednn_lite_config)
endif ()
