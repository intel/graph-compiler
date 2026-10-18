if(NOT GC_ENABLE_PYTHON_BINDINGS)
  message(STATUS "Python bindings are disabled")
  return()
endif()

find_package(
  Python 3.9
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

add_compile_options(-Wno-error=cast-qual -Wno-error=missing-declarations
                    -Wno-switch-default)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wno-error=unused-template -Wno-covered-switch-default)
endif()
nanobind_add_module(graph_compiler Nanobind.cpp)
target_link_libraries(graph_compiler PRIVATE ${PROJECT_NAME})

set_target_properties(
  graph_compiler PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                            ${PROJECT_SOURCE_DIR}/python/src)
