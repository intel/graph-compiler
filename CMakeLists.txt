cmake_minimum_required(VERSION 3.11)
project(graph_compiler)

option(GC_TEST "Enable tests" ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fPIC -ffunction-sections -fdata-sections")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")

set(CMAKE_DEPENDS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(${CMAKE_DEPENDS_DIR}/onednn.cmake)
include(${CMAKE_DEPENDS_DIR}/simdjson.cmake)

include_directories(
        include
        ${dnnl_BINARY_DIR}/include
        ${dnnl_SOURCE_DIR}/include
        ${dnnl_SOURCE_DIR}/src/graph/backend/elyzor/include
        ${simdjson_SOURCE_DIR}/include
)

file(GLOB SOURCES "src/*.*pp")
add_library(graph_compiler SHARED
        ${SOURCES}
)
add_library(graph_compiler_static STATIC
        ${SOURCES}
)

set(LINKED_LIBS
        # dnnl
        simdjson
)

target_link_libraries(graph_compiler PRIVATE ${LINKED_LIBS})
target_link_libraries(graph_compiler_static PRIVATE ${LINKED_LIBS})

add_subdirectory(test)
