include_guard()
include(FetchContent)

# A wrapper around FetchContent that could either fetch content from
# a git repository, use a local directory or find_package().
function(gc_fetch_content
        # The content name. Also used as the package name for find_package().
        name
        # Git tag, branch name or hash. If the value has the format
        # [v]major[.minor[.patch[.tweak]]] and the SKIP_FIND option is not
        # specified, use find_package() first.
        # The variable GC_<NAME>_VERSION could be used to override this argument.
        git_tag_or_version
        # Git repository URL. If the variable GC_<NAME>_SRC_DIR is defined,
        # the local directory is used instead.
        git_repository
        #
        # Optional arguments:
        # SKIP_ADD: Populate but do not add the content to the project.
        # SKIP_FIND: Do not use find_package().
        # SET: key=value variables to be set before the content population.
)
    string(TOUPPER ${name} uname)
    cmake_parse_arguments(GC_${uname} "SKIP_ADD;SKIP_FIND" "" "SET" ${ARGN})

    if (DEFINED GC_${uname}_SET)
        foreach (var ${GC_${uname}_SET})
            string(REGEX REPLACE "([^=]+)=(.*)" "\\1;\\2" var ${var})
            list(GET var 0 key)
            list(GET var 1 value)
            message(STATUS "Setting ${key}=${value}")
            set(${key} ${value})
        endforeach ()
    endif ()

    if (DEFINED GC_${uname}_SRC_DIR)
        FetchContent_Declare(
                ${name}
                SOURCE_DIR ${GC_${uname}_SRC_DIR}
        )
    else ()
        if (DEFINED GC_${uname}_VERSION)
            set(git_tag_or_version ${GC_${uname}_VERSION})
        endif ()
        message(STATUS "${name}_VERSION: ${git_tag_or_version}")

        if (NOT ${uname}_SKIP_FIND AND git_tag_or_version
                MATCHES "^v?([0-9]+(\.[0-9]+(\.[0-9]+(\.[0-9]+)?)?)?)$")
            string(REPLACE "v" "" version ${git_tag_or_version})
            find_package(${name} ${version} EXACT)
        endif ()

        if (NOT DEFINED ${name}_VERSION_COUNT)
            message(STATUS "Fetching ${name} from ${git_repository}")
            set(FETCHCONTENT_QUIET FALSE)
            FetchContent_Declare(
                    ${name}
                    GIT_REPOSITORY ${git_repository}
                    GIT_TAG ${git_tag_or_version}
                    GIT_PROGRESS TRUE
                    FIND_PACKAGE_ARGS ${FIND_PACKAGE_ARGS}
            )
        endif ()
    endif ()

    if (NOT DEFINED ${name}_VERSION_COUNT)
        if (GC_${uname}_SKIP_ADD)
            if (NOT DEFINED ${name}_POPULATED)
                FetchContent_Populate(${name})
                FetchContent_GetProperties(${name})
                set(${name}_POPULATED TRUE PARENT_SCOPE)
            endif ()
        else ()
            FetchContent_MakeAvailable(${name})
        endif ()

        set(${name}_SOURCE_DIR ${${name}_SOURCE_DIR} PARENT_SCOPE)
        set(${name}_BINARY_DIR ${${name}_BINARY_DIR} PARENT_SCOPE)
    endif ()
endfunction()

macro(gc_set_mlir_link_components VAR)
    if(GC_DEV_LINK_LLVM_DYLIB)
        set(${VAR}
            MLIR
        )
    else()
        set(${VAR}
            ${ARGN}
        )
    endif()
endmacro()

function(gc_add_mlir_library name)
    add_mlir_library(${ARGV})

    if(name MATCHES ".+Passes")
        set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS ${name})
    else()
        set_property(GLOBAL APPEND PROPERTY GC_MLIR_LIBS ${name})
    endif()

    if(GcInterface IN_LIST ARGN)
        if(SHARED IN_LIST ARGN)
            target_link_libraries(${name} PUBLIC GcInterface)
        else()
            target_link_libraries(obj.${name} PUBLIC GcInterface)
        endif()
    endif()
endfunction()

function(gc_add_mlir_dialect_library name)
    add_mlir_dialect_library(${ARGV})
    target_link_libraries(obj.${name} PUBLIC GcInterface)
    set_property(GLOBAL APPEND PROPERTY GC_DIALECT_LIBS ${name})

    if(GcInterface IN_LIST ARGN)
        target_link_libraries(obj.${name} PUBLIC GcInterface)
    endif()
endfunction()

macro(gc_add_mlir_tool name)
    # the dependency list copied from mlir/tools/mlir-cpu-runner/CMakeLists.txt of upstream
    if(NOT DEFINED LLVM_LINK_COMPONENTS)
        set(LLVM_LINK_COMPONENTS
          Core
          Support
          nativecodegen
          native
        )
    endif()
    if(NOT DEFINED MLIR_LINK_COMPONENTS)
        gc_set_mlir_link_components(MLIR_LINK_COMPONENTS
          MLIRAnalysis
          MLIRBuiltinToLLVMIRTranslation
          MLIRExecutionEngine
          MLIRIR
          MLIRJitRunner
          MLIRLLVMDialect
          MLIRLLVMToLLVMIRTranslation
          MLIRToLLVMIRTranslationRegistration
          MLIRParser
          MLIRTargetLLVMIRExport
          MLIRSupport
        )
    endif()
    add_mlir_tool(${ARGV})
    #LLVM_LINK_COMPONENTS is processed by LLVM cmake in add_llvm_executable
    target_link_libraries(${name} PRIVATE GcInterface ${MLIR_LINK_COMPONENTS})
    llvm_update_compile_flags(${name})
    set_property(GLOBAL APPEND PROPERTY GC_TOOLS ${name})
endmacro()