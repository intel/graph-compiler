include(onednn)

gc_add_path(GC_LIB_SOURCES GLOB "*.*pp")
gc_add_path(GC_LIB_INCLUDES ${DNNL_INCLUDES})
