if(GC_MLIR_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GC_MLIR_CXX_FLAGS}")
endif()
add_definitions(-DMLIR_INCLUDE_TESTS)
add_custom_target(GCUnitTests)
set_target_properties(GCUnitTests PROPERTIES FOLDER "MLIR GC Tests")

# To silence warning caused by Wundef.
add_definitions(-DGTEST_NO_LLVM_SUPPORT=0)

function(add_mlir_unittest test_dirname)
  add_unittest(GCUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Example)
add_subdirectory(ExecutionEngine)
add_subdirectory(Linalgx)

