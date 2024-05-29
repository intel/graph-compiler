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
        # CMAKE_ARGS: Passed to FetchContent_Declare.
)
    string(TOUPPER ${name} uname)
    cmake_parse_arguments(GC_${uname} "SKIP_ADD;SKIP_FIND" "" "CMAKE_ARGS" ${ARGN})

    if (DEFINED GC_${uname}_CMAKE_ARGS)
        message(STATUS "${name}_CMAKE_ARGS: ${GC_${uname}_CMAKE_ARGS}")
    endif ()

    if (DEFINED GC_${uname}_SRC_DIR)
        FetchContent_Declare(
                ${name}
                SOURCE_DIR ${GC_${uname}_SRC_DIR}
                CMAKE_ARGS ${${uname}_CMAKE_ARGS}
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
                    CMAKE_ARGS ${GC_${uname}_CMAKE_ARGS}
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

# Add one or multiple paths to the specified list.
# The paths could be specified as a list of files or a GLOB pattern:
#   gc_add_path(SOURCES GLOB "src/*.cpp")
#   gc_add_path(INCLUDES include1 include2 include3)
function(gc_add_path list_name paths)
    if (paths STREQUAL "GLOB")
        file(GLOB paths ${ARGN})
        list(APPEND ${list_name} ${paths})
    else ()
        get_filename_component(path ${paths} ABSOLUTE)
        list(APPEND ${list_name} ${path})
        foreach (path ${ARGN})
            get_filename_component(path ${path} ABSOLUTE)
            list(APPEND ${list_name} ${path})
        endforeach ()
    endif ()
    set(${list_name} ${${list_name}}
            CACHE INTERNAL "${list_name} paths"
    )
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
