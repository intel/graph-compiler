include_guard()

if (${GC_GOOGLE_TEST_TAG})
  set(GOOGLETEST_TAG ${GC_GOOGLE_TEST_TAG})
else()
  set(GOOGLETEST_TAG v1.14.0)
endif()
message(STATUS "Using Google Test version: ${GOOGLETEST_TAG}")
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG ${GOOGLETEST_TAG}
  GIT_PROGRESS TRUE
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)