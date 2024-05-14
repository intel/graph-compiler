get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(gc_pass_libs GLOBAL PROPERTY GC_PASS_LIBS)

add_subdirectory(dnnl)
add_subdirectory(gc-cpu-runner)
add_subdirectory(gc-opt)
