if (NOT GC_ENABLE_TEST)
    message(STATUS "The tests are not enabled.")
    return()
endif ()

include(gtest)

