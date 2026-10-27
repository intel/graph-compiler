add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(ExecutionEngine)
add_subdirectory(Runtime)

gc_add_mlir_library(GcGpuCompiler GpuCompiler.cpp LINK_LIBS PUBLIC GcInterface
                    GcGpuPasses)

get_property(GC_MLIR_LIBS GLOBAL PROPERTY GC_MLIR_LIBS)
get_property(GC_PASS_LIBS GLOBAL PROPERTY GC_PASS_LIBS)
get_property(GC_DIALECT_LIBS GLOBAL PROPERTY GC_DIALECT_LIBS)

if(GC_DYLINK)
  set(GC_LIB_OBJECTS)
  foreach(lib IN LISTS GC_MLIR_LIBS GC_PASS_LIBS GC_DIALECT_LIBS)
    list(APPEND GC_LIB_OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endforeach()

  add_library(${PROJECT_NAME} SHARED ${GC_LIB_OBJECTS})
  target_link_libraries(${PROJECT_NAME} PUBLIC GcInterface)
  if(GC_ENABLE_ZE)
    target_link_libraries(${PROJECT_NAME} PRIVATE GcLevelZero)
  endif()
else()
  add_library(${PROJECT_NAME} INTERFACE)
  target_link_libraries(
    ${PROJECT_NAME} INTERFACE GcInterface ${GC_MLIR_LIBS} ${GC_PASS_LIBS}
                              ${GC_DIALECT_LIBS})
endif()
