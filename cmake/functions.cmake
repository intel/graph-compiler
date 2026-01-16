include_guard()

macro(gc_set_mlir_link_components VAR)
    if(GC_DYLINK)
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

function(gc_add_mlir_conversion_library name)
    add_mlir_conversion_library(${ARGV})
    target_link_libraries(obj.${name} PUBLIC GcInterface)
    set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS ${name})

    if(GcInterface IN_LIST ARGN)
        target_link_libraries(obj.${name} PUBLIC GcInterface)
    endif()
endfunction()

function(gc_add_mlir_translation_library name)
    add_mlir_translation_library(${ARGV})
    target_link_libraries(obj.${name} PUBLIC GcInterface)
    set_property(GLOBAL APPEND PROPERTY GC_MLIR_LIBS ${name})

    if(GcInterface IN_LIST ARGN)
        target_link_libraries(obj.${name} PUBLIC GcInterface)
    endif()
endfunction()

macro(gc_add_mlir_tool name)
    add_mlir_tool(${ARGV})
    #LLVM_LINK_COMPONENTS is processed by LLVM cmake in add_llvm_executable
    target_link_libraries(${name} PRIVATE GcInterface ${MLIR_LINK_COMPONENTS})
    llvm_update_compile_flags(${name})
    set_property(GLOBAL APPEND PROPERTY GC_TOOLS ${name})
    set_target_properties(${name} PROPERTIES EXCLUDE_FROM_ALL OFF)
    target_compile_options(${name} PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(${name} PRIVATE -Wl,--gc-sections -Wl,--as-needed)
    mlir_check_all_link_libraries(${name})
endmacro()