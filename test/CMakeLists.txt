if (NOT GC_ENABLE_TEST)
  message(STATUS "Tests are disabled")
  return()
endif()

add_subdirectory(mlir)
