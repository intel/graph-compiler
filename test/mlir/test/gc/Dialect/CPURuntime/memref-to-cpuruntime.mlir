// RUN: gc-opt --split-input-file --convert-memref-to-cpuruntime %s -verify-diagnostics -o -| FileCheck %s

module {
    func.func @doalloc() {
        %m0 = memref.alloc () : memref<13xf32>
        memref.dealloc %m0 : memref<13xf32>
        return
    }
}