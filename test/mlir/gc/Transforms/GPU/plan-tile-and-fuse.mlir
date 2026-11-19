// RUN: gc-opt %s --plan-tile-and-fuse | FileCheck %s

// CHECK-LABEL: func.func @empty() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @empty() {
  return
}
