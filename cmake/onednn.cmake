include_guard()

get_property(GC_DNNL_INCLUDES GLOBAL PROPERTY GC_DNNL_INCLUDES)
if (NOT DEFINED GC_DNNL_INCLUDES)
    # TODO: Change to main https://github.com/oneapi-src/oneDNN.git when all the
    # required functionality is merged.
    gc_fetch_content(dnnl dev https://github.com/kurapov-peter/oneDNN.git
            SKIP_ADD
    )

    set(GC_DNNL_INCLUDES
        $<BUILD_INTERFACE:${dnnl_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${dnnl_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${dnnl_SOURCE_DIR}/src>
    )
    
    set_property(GLOBAL PROPERTY GC_DNNL_INCLUDES ${GC_DNNL_INCLUDES})
    set_property(GLOBAL PROPERTY GC_DNNL_SOURCE_DIR ${dnnl_SOURCE_DIR})

    include(onednn_lite_config)
endif ()
