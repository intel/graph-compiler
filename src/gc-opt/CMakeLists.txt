if(GC_LINK_DYNAMIC_LLVM)
  set(MLIR_LINK_COMPONENTS
    MLIR
  )
else()
  set(MLIR_LINK_COMPONENTS
    MLIROptLib
  )
endif()

set(gc_opt_libs 
        ${dialect_libs}
        ${conversion_libs} 
        ${MLIR_LINK_COMPONENTS} 
        GCPasses)
if(GC_MLIR_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GC_MLIR_CXX_FLAGS}")
endif()

add_llvm_executable(gc-opt gc-opt.cpp)

target_link_libraries(gc-opt PRIVATE ${gc_opt_libs})
llvm_update_compile_flags(gc-opt)
mlir_check_all_link_libraries(gc-opt)

