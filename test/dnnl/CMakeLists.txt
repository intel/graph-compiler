
# Copy resources to the build directory
file(GLOB_RECURSE TEST_RESOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} resources/*)
foreach (TEST_RESOURCE ${TEST_RESOURCES})
    configure_file(${TEST_RESOURCE} ${TEST_RESOURCE} COPYONLY)
endforeach ()

file(GLOB TEST_SOURCES Test*.cpp)
foreach (TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE ${GC_LIB_INCLUDES})
    if (${TEST_NAME} MATCHES "^TestApi.*")
        # The API tests are linked with the shared lib
        target_link_libraries(${TEST_NAME} PRIVATE LLVMSupport graph_compiler)
    else ()
        # The other tests are linked with the static lib and have non-public includes
        target_link_libraries(${TEST_NAME} PRIVATE graph_compiler_static)
        target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src/dnnl)
    endif ()
    target_link_libraries(${TEST_NAME} PRIVATE gtest gtest_main)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach ()
