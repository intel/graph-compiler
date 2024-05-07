set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GC_MLIR_CXX_FLAGS}")
add_subdirectory(Linalgx)
add_subdirectory(Microkernel)
add_subdirectory(OnednnGraph)
