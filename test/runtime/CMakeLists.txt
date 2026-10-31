if(GC_ENABLE_OCL)
  add_executable(gc-ocl-source-test OclSource.cpp)
  target_link_libraries(gc-ocl-source-test PRIVATE GraphCompiler)
  add_test(NAME gc-ocl-source COMMAND gc-ocl-source-test)
  set_tests_properties(gc-ocl-source PROPERTIES SKIP_RETURN_CODE 77)
endif()

add_executable(gc-spirv-test Spirv.cpp)
target_link_libraries(gc-spirv-test PRIVATE GraphCompiler
                                            MLIRSPIRVDeserialization)
add_test(NAME gc-spirv COMMAND gc-spirv-test)
