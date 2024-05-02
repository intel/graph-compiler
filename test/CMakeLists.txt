include(fetch_gtest)
enable_testing()

add_executable(DummyTest DummyTest.cpp)
target_link_libraries(DummyTest PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(DummyTest)