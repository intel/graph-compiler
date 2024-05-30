// RUN: gc-opt %s --convert-cpuruntime-to-llvm --convert-openmp-to-llvm --convert-func-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts | gc-cpu-runner -e main -entry-point-result=void | FileCheck %s
module {
  func.func private @omp_get_thread_num() -> i32

  func.func @check_parallel() {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = llvm.mlir.constant(1 : i64) : i64
    omp.parallel num_threads(%c8: index) {
      omp.wsloop {
        omp.loop_nest (%arg1, %arg2) : index = (%c0, %c0) to (%c1, %c64) step (%c1, %c1) {
          cpuruntime.printf "ITR %zu\n" %arg2 : index
          omp.yield
        }
        omp.terminator
      }
      %tid = func.call @omp_get_thread_num() : () -> i32
      cpuruntime.printf "EXIT %d\n" %tid : i32
      omp.terminator
    }
    return
  }

  func.func @main() {
    %0 = func.call @omp_get_thread_num() : () -> i32
    cpuruntime.printf "TID %d\n" %0 : i32
    call @check_parallel() : ()->()
    return
  }
  // CHECK: TID 0
  // CHECK-COUNT-64: ITR {{[0-9]+}}
  // CHECK-NOT: ITR
  // CHECK-COUNT-8: EXIT {{[0-9]+}}
  // CHECK-NOT: EXIT
}