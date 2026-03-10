add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(ExecutionEngine)

if(GC_DYLINK)
  get_property(GC_MLIR_LIBS GLOBAL PROPERTY GC_MLIR_LIBS)
  get_property(GC_PASS_LIBS GLOBAL PROPERTY GC_PASS_LIBS)
  get_property(GC_DIALECT_LIBS GLOBAL PROPERTY GC_DIALECT_LIBS)

  set(GC_LIB_OBJECTS)
  foreach(lib IN LISTS GC_MLIR_LIBS GC_PASS_LIBS GC_DIALECT_LIBS)
    list(APPEND GC_LIB_OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endforeach()

  add_library(${PROJECT_NAME} SHARED ${GC_LIB_OBJECTS})
  target_link_libraries(${PROJECT_NAME} PUBLIC GcInterface LLVM MLIR
                                               MLIRExecutionEngineShared)
endif()
