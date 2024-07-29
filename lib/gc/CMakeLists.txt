if(GC_MLIR_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GC_MLIR_CXX_FLAGS}")
endif()

include(functions)

add_subdirectory(CAPI)
add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(ExecutionEngine)
add_subdirectory(Target/LLVM)
