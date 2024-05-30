include_guard()

get_property(DNNL_INCLUDES GLOBAL PROPERTY DNNL_INCLUDES)
get_property(DNNL_PATH GLOBAL PROPERTY DNNL_SOURCE_DIR)
if (NOT DEFINED DNNL_INCLUDES)
    return()
endif ()

########## This cmake build lite version of onednn, containing only microkernel related codes

set(APP_NAME "dnnl_brgemm")

# Build onednn
set(DNNL_BUILD_TESTS OFF)
set(DNNL_BUILD_EXAMPLES OFF)
set(DNNL_ENABLE_JIT_PROFILING OFF)
set(DNNL_BLAS_VENDOR NONE)
set(DNNL_LIBRARY_TYPE STATIC)

set(DNNL_GPU_RUNTIME "NONE")
if(NOT DEFINED DNNL_CPU_RUNTIME)
    set(DNNL_CPU_RUNTIME "OMP")
    set(DNNL_CPU_THREADING_RUNTIME "OMP")
endif()

if(${DNNL_CPU_RUNTIME} STREQUAL "OMP")
    find_package(OpenMP REQUIRED)
endif()

if(${DNNL_CPU_RUNTIME} STREQUAL "TBB")
    include("${DNNL_PATH}/cmake/TBB.cmake")
endif()

########## copied from main cmake file of DNNL
# Set the target architecture.
if(NOT DNNL_TARGET_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
        set(DNNL_TARGET_ARCH "AARCH64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64.*|PPC64.*|powerpc64.*)")
        set(DNNL_TARGET_ARCH "PPC64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(s390x.*|S390X.*)")
        set(DNNL_TARGET_ARCH "S390X")
    else()
        set(DNNL_TARGET_ARCH "X64")
    endif()
endif()

if(UNIX OR MINGW)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

########## from cmake/options.cmake
option(DNNL_ENABLE_MAX_CPU_ISA
    "enables control of CPU ISA detected by oneDNN via DNNL_MAX_CPU_ISA
    environment variable and dnnl_set_max_cpu_isa() function" ON)

include("${DNNL_PATH}/cmake/Threading.cmake")

########### copied from cmake/SDL.cmake, for -fstack-protector-strong
if(UNIX)
    set(CMAKE_CCXX_FLAGS "-fPIC -Wformat -Wformat-security -ffunction-sections -fdata-sections")
    append(CMAKE_CXX_FLAGS_RELEASE "-D_FORTIFY_SOURCE=2")
    append(CMAKE_C_FLAGS_RELEASE "-D_FORTIFY_SOURCE=2")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            append(CMAKE_CCXX_FLAGS "-fstack-protector-all")
        else()
            append(CMAKE_CCXX_FLAGS "-fstack-protector-strong")
        endif()

        # GCC might be very paranoid for partial structure initialization, e.g.
        #   struct { int a, b; } s = { 0, };
        # However the behavior is triggered by `Wmissing-field-initializers`
        # only. To prevent warnings on users' side who use the library and turn
        # this warning on, let's use it too. Applicable for the library sources
        # and interfaces only (tests currently rely on that fact heavily)
        append(CMAKE_SRC_CCXX_FLAGS "-Wmissing-field-initializers")
        append(CMAKE_EXAMPLE_CCXX_FLAGS "-Wmissing-field-initializers")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        append(CMAKE_CCXX_FLAGS "-fstack-protector-all")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        append(CMAKE_CXX_FLAGS "-fstack-protector")
    endif()
    append(CMAKE_C_FLAGS "${CMAKE_CCXX_FLAGS}")
    append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_FLAGS}")
    if(APPLE)
        append(CMAKE_SHARED_LINKER_FLAGS "-Wl,-bind_at_load")
        append(CMAKE_EXE_LINKER_FLAGS "-Wl,-bind_at_load")
    else()
        append(CMAKE_EXE_LINKER_FLAGS "-pie")
        append(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
        append(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
    endif()
elseif(MSVC AND ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CMAKE_CCXX_FLAGS "/guard:cf")
endif()
########### END of copy of cmake/SDL.cmake

########### copied from cmake/platform.cmake, for STDC* and -msse4.1
add_definitions(-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS)
if(MSVC)
    set(USERCONFIG_PLATFORM "x64")
    append_if(DNNL_WERROR CMAKE_CCXX_FLAGS "/WX")
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        append(CMAKE_CCXX_FLAGS "/MP")
        # int -> bool
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4800")
        # unknown pragma
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4068")
        # double -> float
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4305")
        # UNUSED(func)
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4551")
        # int64_t -> int (tent)
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4244")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        append(CMAKE_CCXX_FLAGS "/MP")
        set(DEF_ARCH_OPT_FLAGS "-QxSSE4.1")
        # disable: loop was not vectorized with "simd"
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:13379")
        # disable: loop was not vectorized with "simd"
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:15552")
        # disable: unknown pragma
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:3180")
        # disable: foo has been targeted for automatic cpu dispatch
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:15009")
        # disable: disabling user-directed function packaging (COMDATs)
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:11031")
        # disable: decorated name length exceeded, name was truncated
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:2586")
        # disable: disabling optimization; runtime debug checks enabled
        append(CMAKE_CXX_FLAGS_DEBUG "-Qdiag-disable:10182")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        append(CMAKE_CCXX_NOEXCEPT_FLAGS "-fno-exceptions")
        # Clang cannot vectorize some loops with #pragma omp simd and gets
        # very upset. Tell it that it's okay and that we love it
        # unconditionally.
        append(CMAKE_CCXX_FLAGS "-Wno-pass-failed")
        # Clang doesn't like the idea of overriding optimization flags.
        # We don't want to optimize jit gemm kernels to reduce compile time
        append(CMAKE_CCXX_FLAGS "-Wno-overriding-t-option")
    endif()
elseif(UNIX OR MINGW)
    append(CMAKE_CCXX_FLAGS "-Wall -Wno-unknown-pragmas")
    append_if(DNNL_WERROR CMAKE_CCXX_FLAGS "-Werror")
    append(CMAKE_CCXX_FLAGS "-fvisibility=internal")
    append(CMAKE_CXX_FLAGS "-fvisibility-inlines-hidden")
    append(CMAKE_CCXX_NOEXCEPT_FLAGS "-fno-exceptions")
    # compiler specific settings
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(DNNL_TARGET_ARCH STREQUAL "AARCH64")
             set(DEF_ARCH_OPT_FLAGS "-O3")
             # For native compilation tune for the host processor
             if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                 append(DEF_ARCH_OPT_FLAGS "-mcpu=native")
             endif()
        elseif(DNNL_TARGET_ARCH STREQUAL "PPC64")
             set(DEF_ARCH_OPT_FLAGS "-O3")
             # For native compilation tune for the host processor
             if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                 append(DEF_ARCH_OPT_FLAGS "-mcpu=native")
             endif()
        elseif(DNNL_TARGET_ARCH STREQUAL "S390X")
             set(DEF_ARCH_OPT_FLAGS "-O3")
             # For native compilation tune for the host processor
             if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                 append(DEF_ARCH_OPT_FLAGS "-march=native")
             endif()
        elseif(DNNL_TARGET_ARCH STREQUAL "X64")
             set(DEF_ARCH_OPT_FLAGS "-msse4.1")
        endif()
        # Clang cannot vectorize some loops with #pragma omp simd and gets
        # very upset. Tell it that it's okay and that we love it
        # unconditionally.
        append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-pass-failed")
        if(DNNL_USE_CLANG_SANITIZER MATCHES "Memory(WithOrigin)?")
            if(NOT DNNL_CPU_THREADING_RUNTIME STREQUAL "SEQ")
                message(WARNING "Clang OpenMP is not compatible with MSan! "
                    "Expect a lot of false positives!")
            endif()
            append(CMAKE_CCXX_SANITIZER_FLAGS "-fsanitize=memory")
            if(DNNL_USE_CLANG_SANITIZER STREQUAL "MemoryWithOrigin")
                append(CMAKE_CCXX_SANITIZER_FLAGS
                    "-fsanitize-memory-track-origins=2")
                append(CMAKE_CCXX_SANITIZER_FLAGS
                    "-fno-omit-frame-pointer")
            endif()
            set(DNNL_ENABLED_CLANG_SANITIZER "${DNNL_USE_CLANG_SANITIZER}")
        elseif(DNNL_USE_CLANG_SANITIZER STREQUAL "Undefined")
            append(CMAKE_CCXX_SANITIZER_FLAGS "-fsanitize=undefined")
            append(CMAKE_CCXX_SANITIZER_FLAGS
                "-fno-sanitize=function,vptr")  # work around linking problems
            append(CMAKE_CCXX_SANITIZER_FLAGS "-fno-omit-frame-pointer")
            set(DNNL_ENABLED_CLANG_SANITIZER "${DNNL_USE_CLANG_SANITIZER}")
        elseif(DNNL_USE_CLANG_SANITIZER STREQUAL "Address")
            append(CMAKE_CCXX_SANITIZER_FLAGS "-fsanitize=address")
            set(DNNL_ENABLED_CLANG_SANITIZER "${DNNL_USE_CLANG_SANITIZER}")
        elseif(DNNL_USE_CLANG_SANITIZER STREQUAL "Thread")
            append(CMAKE_CCXX_SANITIZER_FLAGS "-fsanitize=thread")
            set(DNNL_ENABLED_CLANG_SANITIZER "${DNNL_USE_CLANG_SANITIZER}")
        elseif(DNNL_USE_CLANG_SANITIZER STREQUAL "Leak")
            append(CMAKE_CCXX_SANITIZER_FLAGS "-fsanitize=leak")
            set(DNNL_ENABLED_CLANG_SANITIZER "${DNNL_USE_CLANG_SANITIZER}")
        elseif(NOT DNNL_USE_CLANG_SANITIZER STREQUAL "")
            message(FATAL_ERROR
                "Unsupported Clang sanitizer '${DNNL_USE_CLANG_SANITIZER}'")
        endif()
        if(DNNL_ENABLED_CLANG_SANITIZER)
            message(STATUS
                "Using Clang ${DNNL_ENABLED_CLANG_SANITIZER} "
                "sanitizer (experimental!)")
            append(CMAKE_CCXX_SANITIZER_FLAGS "-g -fno-omit-frame-pointer")
        endif()

        if (DNNL_USE_CLANG_TIDY MATCHES "(CHECK|FIX)" AND ${CMAKE_VERSION} VERSION_LESS "3.6.0")
            message(FATAL_ERROR "Using clang-tidy requires CMake 3.6.0 or newer")
        elseif(DNNL_USE_CLANG_TIDY MATCHES "(CHECK|FIX)")
            find_program(CLANG_TIDY NAMES clang-tidy)
            if(NOT CLANG_TIDY)
                message(FATAL_ERROR "Clang-tidy not found")
            else()
                if(DNNL_USE_CLANG_TIDY STREQUAL "CHECK")
                    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY})
                    message(STATUS "Using clang-tidy to run checks")
                elseif(DNNL_USE_CLANG_TIDY STREQUAL "FIX")
                    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY} -fix)
                    message(STATUS "Using clang-tidy to run checks and fix found issues")
                endif()
            endif()
        endif()

    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if(DNNL_TARGET_ARCH STREQUAL "AARCH64")
             set(DEF_ARCH_OPT_FLAGS "-O3")
             # For native compilation tune for the host processor
             if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                 append(DEF_ARCH_OPT_FLAGS "-mcpu=native")
             endif()
        elseif(DNNL_TARGET_ARCH STREQUAL "PPC64")
             set(DEF_ARCH_OPT_FLAGS "-O3")
             # In GCC, -ftree-vectorize is turned on under -O3 since 2007.
             # For native compilation tune for the host processor
             if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                 append(DEF_ARCH_OPT_FLAGS "-mcpu=native")
             endif()
        elseif(DNNL_TARGET_ARCH STREQUAL "S390X")
             set(DEF_ARCH_OPT_FLAGS "-O3")
             # In GCC, -ftree-vectorize is turned on under -O3 since 2007.
             # For native compilation tune for the host processor
             if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                 append(DEF_ARCH_OPT_FLAGS "-march=native")
             endif()
        elseif(DNNL_TARGET_ARCH STREQUAL "X64")
             set(DEF_ARCH_OPT_FLAGS "-msse4.1")
        endif()
        # suppress warning on assumptions made regarding overflow (#146)
        append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-strict-overflow")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(DEF_ARCH_OPT_FLAGS "-xSSE4.1")
        # workaround for Intel Compiler that produces error caused
        # by pragma omp simd collapse(..)
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:13379")
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:15552")
        # disable `was not vectorized: vectorization seems inefficient` remark
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:15335")
        # disable: foo has been targeted for automatic cpu dispatch
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:15009")
    endif()
endif()

append(CMAKE_C_FLAGS "${CMAKE_CCXX_FLAGS} ${DEF_ARCH_OPT_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_FLAGS} ${DEF_ARCH_OPT_FLAGS}")

########### END of copy of cmake/platform.cmake

########### setting dummy version info
set(DNNL_VERSION_MAJOR 0)
set(DNNL_VERSION_MINOR 0)
set(DNNL_VERSION_PATCH 0)
set(DNNL_VERSION_HASH "N/A")
########### END of setting dummy version info

add_definitions(-DDNNL_ENABLE_JIT_PROFILING=0)
configure_file(
    "${DNNL_PATH}/include/oneapi/dnnl/dnnl_config.h.in"
    "${PROJECT_BINARY_DIR}/include/oneapi/dnnl/dnnl_config.h"
)

configure_file(
    "${DNNL_PATH}/include/oneapi/dnnl/dnnl_version.h.in"
    "${PROJECT_BINARY_DIR}/include/oneapi/dnnl/dnnl_version.h"
)

include_directories(
    ${PROJECT_BINARY_DIR}/include
    ${DNNL_PATH}/src
    ${DNNL_PATH}/include
    )

if(DNNL_ENABLE_MAX_CPU_ISA)
    add_definitions(-DDNNL_ENABLE_MAX_CPU_ISA)
endif()

add_definitions(-DDISABLE_VERBOSE=1)
file(GLOB_RECURSE DNNL_SOURCES
    ${DNNL_PATH}/src/cpu/x64/brgemm/*.cpp
    ${DNNL_PATH}/src/cpu/x64/injectors/*.cpp
    ${DNNL_PATH}/src/cpu/x64/cpu_isa_traits.cpp
    ${DNNL_PATH}/src/cpu/x64/jit_avx512_core_bf16cvt.cpp
    ${DNNL_PATH}/src/cpu/x64/amx_tile_configure.[ch]pp
    ${DNNL_PATH}/src/cpu/x64/jit_uni_convert_xf16.[ch]pp
    ${DNNL_PATH}/src/cpu/jit_utils/jit_utils.cpp
    ${DNNL_PATH}/src/cpu/platform.[ch]pp
    ${DNNL_PATH}/src/cpu/bfloat16.cpp
    ${DNNL_PATH}/src/cpu/binary_injector_utils.cpp
    ${DNNL_PATH}/src/common/fpmath_mode.cpp
    ${DNNL_PATH}/src/common/utils.cpp
    ${DNNL_PATH}/src/common/bfloat16.[ch]pp
    ${DNNL_PATH}/src/common/float8.[ch]pp
    ${DNNL_PATH}/src/common/memory_debug.cpp
    ${DNNL_PATH}/src/common/primitive_attr.cpp
    ${DNNL_PATH}/src/common/broadcast_strategy.cpp
    ${DNNL_PATH}/src/common/primitive_exec_types.cpp
    ${DNNL_PATH}/src/common/memory.cpp
    ${DNNL_PATH}/src/common/memory_zero_pad.cpp
    ${DNNL_PATH}/src/common/memory_desc_wrapper.cpp
    ${DNNL_PATH}/src/common/memory_desc.cpp
    ${DNNL_PATH}/src/common/dnnl_thread.cpp
    ${DNNL_PATH}/src/common/verbose.cpp
    ${DNNL_PATH}/src/common/dnnl_debug.cpp
    ${DNNL_PATH}/src/common/dnnl_debug_autogenerated.cpp
    ${DNNL_PATH}/src/cpu/x64/jit_avx512_core_fp8cvt.cpp
    )

add_library(dnnl_brgemm OBJECT ${DNNL_SOURCES})
set_property(TARGET dnnl_brgemm PROPERTY POSITION_INDEPENDENT_CODE ON
             CXX_VISIBILITY_PRESET "hidden"
             VISIBILITY_INLINES_HIDDEN 1)

# install(TARGETS dnnl_brgemm
#        EXPORT dnnl_brgemm_export
#        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# set_property(GLOBAL APPEND PROPERTY DNNL_SUBDIR_EXTRA_STATIC_LIBS $<BUILD_INTERFACE:dnnl_brgemm>)
# set_property(GLOBAL APPEND PROPERTY DNNL_SUBDIR_EXTRA_SHARED_LIBS dnnl_brgemm)
# Currently build objs only
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:dnnl_brgemm>)
