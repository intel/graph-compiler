if(NOT GC_ENABLE_TEST)
  message(STATUS "Tests are disabled")
  return()
endif()

if(GC_ENABLE_IGC)
  add_subdirectory(mlir)
endif()
add_subdirectory(runtime)
