find_path(LevelZero_INCLUDE_DIR NAMES level_zero/ze_api.h)
find_library(LevelZero_LIBRARY NAMES ze_loader)
if(NOT LevelZero_INCLUDE_DIR OR NOT LevelZero_LIBRARY)
  message(FATAL_ERROR "Level Zero (ze_loader) not found")
endif()

add_library(GcLevelZero INTERFACE)
target_include_directories(GcLevelZero INTERFACE "${LevelZero_INCLUDE_DIR}")
target_link_libraries(GcLevelZero INTERFACE "${LevelZero_LIBRARY}")

gc_add_mlir_library(GcZeGpuRuntime ZeGpuRuntime.cpp LINK_LIBS PUBLIC
                    GcInterface)
target_link_libraries(GcZeGpuRuntime PRIVATE GcLevelZero)
