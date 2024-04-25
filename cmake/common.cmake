include_guard()
include(FetchContent)

function(DeclareContent name version url)
    string(TOUPPER ${name} ${name}_UPPER)
    set(${${name}_UPPER}_CMAKE_ARGS "-DCMAKE_CXX_FLAGS=\"${CMAKE_CXX_FLAGS}\"" ${${${name}_UPPER}_CMAKE_ARGS})
    message(STATUS "${name} cmake args: ${${${name}_UPPER}_CMAKE_ARGS}")

    if (DEFINED ${${name}_UPPER}_SRC_DIR)
        FetchContent_Declare(
                ${name}
                SOURCE_DIR ${${${name}_UPPER}_SRC_DIR}
                CMAKE_ARGS ${${${name}_UPPER}_CMAKE_ARGS}
        )
    else ()
        if (NOT DEFINED ${${name}_UPPER}_VERSION)
            set(${${name}_UPPER}_VERSION ${version})
        endif ()

        message(STATUS "${name} version: ${${${name}_UPPER}_VERSION}")
        FetchContent_Declare(
                ${name}
                GIT_REPOSITORY ${url}
                GIT_TAG ${${${name}_UPPER}_VERSION}
                CMAKE_ARGS ${CMAKE_CXX_FLAGS} ${${${name}_UPPER}_CMAKE_ARGS}
        )
    endif ()
endfunction()
