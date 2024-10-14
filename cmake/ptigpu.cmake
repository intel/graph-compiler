include_guard()

FetchContent_Declare(
  PTIGPU
  GIT_REPOSITORY https://github.com/zhczhong/pti-gpu.git
  GIT_TAG        master
  SOURCE_SUBDIR  tools/onetrace
)
FetchContent_MakeAvailable(PTIGPU)

set_property(GLOBAL PROPERTY GC_PTIGPU_BINARY_DIR ${ptigpu_BINARY_DIR})

target_compile_options(GcInterface INTERFACE -DGC_ENABLE_GPU_PROFILE)

