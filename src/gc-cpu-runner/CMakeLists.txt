set(LLVM_LINK_COMPONENTS
  LLVM
  )
set(gc_cpu_runner_libs 
        ${dialect_libs}
        ${conversion_libs} 
        ${LLVM_LINK_COMPONENTS}
        MLIR
        MLIRExecutionEngine
        MLIRJitRunner
        GCPasses)
add_llvm_executable(gc-cpu-runner gc-cpu-runner.cpp)

target_link_libraries(gc-cpu-runner PRIVATE ${gc_cpu_runner_libs})
llvm_update_compile_flags(gc-cpu-runner)
mlir_check_all_link_libraries(gc-cpu-runner)
