if(GC_LINK_DYNAMIC_LLVM)
  set(LLVM_LINK_COMPONENTS
    LLVM
    )
  set(MLIR_LINK_COMPONENTS
    MLIR
    MLIRExecutionEngine
    MLIRJitRunner
  )
else()
  if(NOT LLVM_LINK_LLVM_DYLIB)
    set(LLVM_LINK_COMPONENTS
      Core
      Support
      nativecodegen
      native
    )
  endif()
  set(MLIR_LINK_COMPONENTS
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

if(GC_MLIR_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GC_MLIR_CXX_FLAGS}")
endif()

#LLVM_LINK_COMPONENTS is processed by LLVM cmake in add_llvm_executable
set(gc_cpu_runner_libs
        ${MLIR_LINK_COMPONENTS})
add_llvm_executable(gc-cpu-runner gc-cpu-runner.cpp)

target_link_libraries(gc-cpu-runner PRIVATE ${gc_cpu_runner_libs})
llvm_update_compile_flags(gc-cpu-runner)
mlir_check_all_link_libraries(gc-cpu-runner)
