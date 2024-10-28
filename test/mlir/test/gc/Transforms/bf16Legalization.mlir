// RUN: gc-opt %s --math-extend-to-supported-types --arith-emulate-unsupported-floats="source-types=bf16 target-type=f32" --canonicalize | FileCheck %s

// CHECK-LABEL: @sin
// CHECK-SAME: ([[ARG0:%.+]]: bf16)
func.func @sin(%arg0: bf16) -> bf16 {
  // CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
  // CHECK: [[SIN:%.+]] = math.sin [[EXTF]]
  // CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
  // CHECK: return [[TRUNCF]] : bf16
  %0 = math.sin %arg0 : bf16
  return %0 : bf16
}

// CHECK-LABEL: @abs_vector
// CHECK-SAME: ([[ARG0:%.+]]: vector<2xf16>)
// CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABS:%.+]] = math.absf [[EXTF]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[ABS]]
// CHECK: return [[TRUNCF]] : vector<2xf16>
func.func @abs_vector(%arg0: vector<2xf16>) -> vector<2xf16> {
  %0 = math.absf %arg0 : vector<2xf16>
  return %0 : vector<2xf16>
}

// CHECK-LABEL: @sequences_bf16
// CHECK-SAME: ([[ARG0:%.+]]: bf16)
// CHECK: [[EXTF0:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABSF:%.+]] = math.absf [[EXTF0]]
// CHECK: [[SIN:%.+]] = math.sin [[ABSF]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
// CHECK: return [[TRUNCF]] : bf16
func.func @sequences_bf16(%arg0: bf16) -> bf16 {
  %0 = math.absf %arg0 : bf16
  %1 = math.sin %0 : bf16
  return %1 : bf16
}

// CHECK-LABEL: @bf16_branch_vector
// CHECK-SAME: ([[ARG0:%.+]]: vector<2xbf16>)
// CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABSF:%.+]] = math.absf [[EXTF]]
// CHECK: [[SIN:%.+]] = math.sin [[ABSF]]
// CHECK: [[COS:%.+]] = math.cos [[ABSF]]
// CHECK: [[ADDF:%.+]] = arith.addf
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[ADDF]]
// CHECK: return [[TRUNCF]] : vector<2xbf16>
func.func @bf16_branch_vector(%arg0: vector<2xbf16>) -> vector<2xbf16> {
  %0 = math.absf %arg0 : vector<2xbf16>
  %1 = math.sin %0 : vector<2xbf16>
  %2 = math.cos %0 : vector<2xbf16>
  %3 = arith.addf %1, %2 : vector<2xbf16>
  return %3 : vector<2xbf16>
}
