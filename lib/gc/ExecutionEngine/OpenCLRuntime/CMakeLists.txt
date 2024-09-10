find_package(OpenCL REQUIRED)

gc_add_mlir_library(GcOpenclRuntime
    SHARED
    OpenCLRuntimeWrappers.cpp

    EXCLUDE_FROM_LIBMLIR
  )

check_cxx_compiler_flag("-frtti" CXX_HAS_FRTTI_FLAG)
if(NOT CXX_HAS_FRTTI_FLAG)
    message(FATAL_ERROR "CXX compiler does not accept flag -frtti")
endif()
target_compile_options (GcOpenclRuntime PUBLIC -fexceptions -frtti)

target_include_directories(GcOpenclRuntime PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    )

message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
target_link_libraries(GcOpenclRuntime PUBLIC ${OpenCL_LIBRARIES})
