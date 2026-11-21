# Security hardening flags, following the policy OpenVINO implements in
# cmake/developer_package/compile_flags/sdl.cmake.
#
# The flags are added to the Release configuration only, as OpenVINO does on
# Linux: _FORTIFY_SOURCE needs an optimizing build to have any effect, and the
# Debug configuration is not shipped. Every flag is probed before use, so an
# older or unusual toolchain loses the flag instead of failing the build.

include_guard()

option(GC_ENABLE_SDL "Enable the security hardening flags" ON)

if(NOT GC_ENABLE_SDL)
  message(
    WARNING "GC_ENABLE_SDL=OFF: the Release build is not hardened and must not "
            "be shipped.")
  return()
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
  message(WARNING "GC_ENABLE_SDL: no hardening flags are defined for compiler "
                  "'${CMAKE_CXX_COMPILER_ID}'.")
  return()
endif()

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(CheckLinkerFlag)

set(gc_sdl_compile_options -Wformat -Wformat-security)

# Raise _FORTIFY_SOURCE to at least 2, but never lower what the toolchain
# already provides. GCC on Ubuntu predefines _FORTIFY_SOURCE=3 when optimizing,
# and passing -D_FORTIFY_SOURCE=2 would silently weaken it. The probe runs with
# the Release flags because GCC only predefines the macro when optimizing.
set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
check_cxx_source_compiles(
  "#if !defined(_FORTIFY_SOURCE) || _FORTIFY_SOURCE < 2
   #error _FORTIFY_SOURCE is weaker than 2
   #endif
   int main() {}"
  GC_HAS_FORTIFY_SOURCE)
unset(CMAKE_REQUIRED_FLAGS)
if(NOT GC_HAS_FORTIFY_SOURCE)
  list(APPEND gc_sdl_compile_options -D_FORTIFY_SOURCE=2)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND gc_sdl_compile_options -fstack-protector-strong
       -fno-strict-overflow -fno-delete-null-pointer-checks -fwrapv)
else()
  list(APPEND gc_sdl_compile_options -fstack-protector-all)
endif()

foreach(flag IN LISTS gc_sdl_compile_options)
  string(MAKE_C_IDENTIFIER "GC_HAS_${flag}" flag_var)
  check_cxx_compiler_flag(${flag} ${flag_var})
  if(${flag_var})
    string(APPEND CMAKE_C_FLAGS_RELEASE " ${flag}")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " ${flag}")
  else()
    message(STATUS "GC_ENABLE_SDL: the compiler does not support ${flag}")
  endif()
endforeach()

# A non-executable stack, and relocations that are read-only and resolved at
# load time, for every binary.
foreach(flag "-z noexecstack" "-z relro" "-z now")
  string(MAKE_C_IDENTIFIER "GC_HAS_LINK_${flag}" flag_var)
  check_linker_flag(CXX ${flag} ${flag_var})
  if(${flag_var})
    string(APPEND CMAKE_SHARED_LINKER_FLAGS_RELEASE " ${flag}")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS_RELEASE " ${flag}")
    string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " ${flag}")
  else()
    message(STATUS "GC_ENABLE_SDL: the linker does not support ${flag}")
  endif()
endforeach()

# Position independent executables. The objects are already position
# independent, because LLVM builds with -fPIC by default.
check_linker_flag(CXX -pie GC_HAS_LINK_pie)
if(GC_HAS_LINK_pie)
  string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " -pie")
else()
  message(STATUS "GC_ENABLE_SDL: the linker does not support -pie")
endif()
