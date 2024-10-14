include_guard()

FetchContent_Declare(
  PTIGPU
  GIT_REPOSITORY https://github.com/intel/pti-gpu.git
  GIT_TAG        exp_opencl_0.11.0
  SOURCE_SUBDIR  sdk
)
FetchContent_MakeAvailable(PTIGPU)
