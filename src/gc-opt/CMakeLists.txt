if(GC_DEV_LINK_LLVM_DYLIB)
  set(MLIR_LINK_COMPONENTS
    MLIR
  )
  get_property(dialect_libs GLOBAL PROPERTY GC_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY GC_PASS_LIBS)
else()
  set(MLIR_LINK_COMPONENTS
    MLIROptLib
  )
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
endif()

set(gc_opt_libs 
        ${dialect_libs}
        ${conversion_libs} 
        ${MLIR_LINK_COMPONENTS} 
        GCPasses
        GCAnalysis
        GCCpuRuntime)

if(GC_MLIR_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GC_MLIR_CXX_FLAGS}")
endif()

add_llvm_executable(gc-opt gc-opt.cpp)

target_link_libraries(gc-opt PRIVATE ${gc_opt_libs})
llvm_update_compile_flags(gc-opt)
mlir_check_all_link_libraries(gc-opt)

