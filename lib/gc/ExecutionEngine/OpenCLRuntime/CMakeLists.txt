find_package(OpenCL REQUIRED)

add_mlir_library(mlir_opencl_runtime
    SHARED
    OpenCLRuntimeWrappers.cpp

    EXCLUDE_FROM_LIBMLIR
  )

check_cxx_compiler_flag("-frtti" CXX_HAS_FRTTI_FLAG)
if(NOT CXX_HAS_FRTTI_FLAG)
    message(FATAL_ERROR "CXX compiler does not accept flag -frtti")
endif()
target_compile_options (mlir_opencl_runtime PUBLIC -fexceptions -frtti)

target_include_directories(mlir_opencl_runtime PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
    )

message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
target_link_libraries(mlir_opencl_runtime PUBLIC ${OpenCL_LIBRARIES})
