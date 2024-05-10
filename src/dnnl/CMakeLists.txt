include(onednn)

gc_add_path(GC_LIB_SOURCES GLOB "*.cpp")
gc_add_path(GC_LIB_INCLUDES ${DNNL_INCLUDES})
