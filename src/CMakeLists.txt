get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(gc_opt_libs 
        ${dialect_libs}
        ${conversion_libs} 
        MLIROptLib 
        GCPasses)
add_llvm_executable(gc-opt gc-opt.cpp)

target_link_libraries(gc-opt PRIVATE ${gc_opt_libs})
llvm_update_compile_flags(gc-opt)
mlir_check_all_link_libraries(gc-opt)

add_subdirectory(dnnl)
