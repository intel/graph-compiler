find_package(OpenCL REQUIRED)

if(NOT OpenCL_FOUND)
    message(FATAL_ERROR "OpenCL not found.")
endif()

add_mlir_library(opencl-runtime
    SHARED
    OpenCLRuntimeWrappers.cpp

    EXCLUDE_FROM_LIBMLIR
  )

check_cxx_compiler_flag("-frtti" CXX_HAS_FRTTI_FLAG)
if(NOT CXX_HAS_FRTTI_FLAG)
    message(FATAL_ERROR "CXX compiler does not accept flag -frtti")
endif()
target_compile_options (opencl-runtime PUBLIC -fexceptions -frtti)

target_include_directories(opencl-runtime PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
    )

message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
target_link_libraries(opencl-runtime PRIVATE ${OpenCL_LIBRARIES})
