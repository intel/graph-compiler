# Copy resources to the build directory
file(GLOB_RECURSE TEST_RESOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} resources/*)
foreach (TEST_RESOURCE ${TEST_RESOURCES})
    configure_file(${TEST_RESOURCE} ${TEST_RESOURCE} COPYONLY)
endforeach ()

file(GLOB TEST_SOURCES test_*.cpp)

foreach (TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PUBLIC ${GC_LIB_INCLUDES})
    target_link_libraries(${TEST_NAME} graph_compiler_static gtest gtest_main)
    add_test(NAME ${TEST_NAME}
            COMMAND ${TEST_NAME}
    )
endforeach ()
