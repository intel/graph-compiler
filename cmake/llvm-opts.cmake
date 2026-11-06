# Included by scripts/build.sh via CMAKE_PROJECT_LLVM_INCLUDE.
if(MSVC)
  set(opt "/O1 /Gw /DNDEBUG")
else()
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-Oz GC_HAS_OZ)
  if(GC_HAS_OZ)
    set(opt "-Oz -DNDEBUG")
  else()
    set(opt "-Os -DNDEBUG")
  endif()
  set(CMAKE_CXX_FLAGS_DEBUG
      "-g -O0"
      CACHE STRING "" FORCE)
endif()
set(CMAKE_C_FLAGS_RELEASE
    "${opt}"
    CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE
    "${opt}"
    CACHE STRING "" FORCE)
