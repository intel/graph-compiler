// RUN: gc-opt %s --split-input-file --legalizedtype-to-f32 | FileCheck %s

// CHECK-LABEL: @sin_bf16
// CHECK-SAME: ([[ARG0:%.+]]: bf16)
func.func @sin_bf16(%arg0: bf16) -> bf16 {
  // CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
  // CHECK: [[SIN:%.+]] = math.sin [[EXTF]]
  // CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
  // CHECK: return [[TRUNCF]] : bf16
  %0 = math.sin %arg0 : bf16
  return %0 : bf16
}

// CHECK-LABEL: @sin_f16
// CHECK-SAME: ([[ARG0:%.+]]: f16)
func.func @sin_f16(%arg0: f16) -> f16 {
  // CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
  // CHECK: [[SIN:%.+]] = math.sin [[EXTF]]
  // CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
  // CHECK: return [[TRUNCF]] : f16
  %0 = math.sin %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @abs_bf16
// CHECK-SAME: ([[ARG0:%.+]]: bf16)
func.func @abs_bf16(%arg0: bf16) -> bf16 {
  // CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
  // CHECK: [[ABS:%.+]] = math.absf [[EXTF]]
  // CHECK: [[TRUNCF:%.+]] = arith.truncf [[ABS]]
  // CHECK: return [[TRUNCF]] : bf16
  %0 = math.absf %arg0 : bf16
  return %0 : bf16
}

// COM: Verify that the pass leaves `math.absf` with `float16` untouched, since the default
// COM: cpuflags.fAVX512FP16 is true. 
// COM: May change this test case when target machine desciption is ready.
// CHECK-LABEL: @abs_f16
// CHECK-SAME: ([[ARG0:%.+]]: f16)
func.func @abs_f16(%arg0: f16) -> f16 {
  // CHECK: [[ABS:%.+]] = math.absf [[ARG0]]
  // CHECK: return [[ABS]] : f16
  %0 = math.absf %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @fma_bf16
// CHECK-SAME: ([[ARG0:%.+]]: bf16, [[ARG1:%.+]]: bf16, [[ARG2:%.+]]: bf16)
// CHECK: [[EXTF0:%.+]] = arith.extf [[ARG0]]
// CHECK: [[EXTF1:%.+]] = arith.extf [[ARG1]]
// CHECK: [[EXTF2:%.+]] = arith.extf [[ARG2]]
// CHECK: [[FMA:%.+]] = math.fma [[EXTF0]], [[EXTF1]], [[EXTF2]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[FMA]]
// CHECK: return [[TRUNCF]] : bf16
func.func @fma_bf16(%arg0: bf16, %arg1: bf16, %arg2: bf16) -> bf16 {
  %0 = math.fma %arg0, %arg1, %arg2 : bf16
  return %0 : bf16
}

// CHECK-LABEL: @fma_f16
// CHECK-SAME: ([[ARG0:%.+]]: f16, [[ARG1:%.+]]: f16, [[ARG2:%.+]]: f16)
// CHECK: [[EXTF0:%.+]] = arith.extf [[ARG0]]
// CHECK: [[EXTF1:%.+]] = arith.extf [[ARG1]]
// CHECK: [[EXTF2:%.+]] = arith.extf [[ARG2]]
// CHECK: [[FMA:%.+]] = math.fma [[EXTF0]], [[EXTF1]], [[EXTF2]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[FMA]]
// CHECK: return [[TRUNCF]] : f16
func.func @fma_f16(%arg0: f16, %arg1: f16, %arg2: f16) -> f16 {
  %0 = math.fma %arg0, %arg1, %arg2 : f16
  return %0 : f16
}

// CHECK-LABEL: @absf_f64
// CHECK-SAME: ([[ARG0:%.+]]: f64)
// CHECK: [[ABSF:%.+]] = math.absf [[ARG0]]
// CHECK: return [[ABSF]] : f64
func.func @absf_f64(%arg0: f64) -> f64 {
  %0 = math.absf %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @sin_vector
// CHECK-SAME: ([[ARG0:%.+]]: vector<2xbf16>)
// CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
// CHECK: [[SIN:%.+]] = math.sin [[EXTF]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
// CHECK: return [[TRUNCF]] : vector<2xbf16>
func.func @sin_vector(%arg0: vector<2xbf16>) -> vector<2xbf16> {
  %0 = math.sin %arg0 : vector<2xbf16>
  return %0 : vector<2xbf16>
}

// CHECK-LABEL: @abs_vector
// CHECK-SAME: ([[ARG0:%.+]]: vector<2xf16>)
// CHECK: [[ABS:%.+]] = math.absf [[ARG0]]
// CHECK: return [[ABS]] : vector<2xf16>
func.func @abs_vector(%arg0: vector<2xf16>) -> vector<2xf16> {
  %0 = math.absf %arg0 : vector<2xf16>
  return %0 : vector<2xf16>
}

// CHECK-LABEL: @sequences_bf16
// CHECK-SAME: ([[ARG0:%.+]]: bf16)
// CHECK: [[EXTF0:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABSF:%.+]] = math.absf [[EXTF0]]
// CHECK: [[TRUNCF0:%.+]] = arith.truncf [[ABSF]]
// CHECK: [[EXTF1:%.+]] = arith.extf [[TRUNCF0]]
// CHECK: [[SIN:%.+]] = math.sin [[EXTF1]]
// CHECK: [[TRUNCF1:%.+]] = arith.truncf [[SIN]]
// CHECK: return [[TRUNCF1]] : bf16
func.func @sequences_bf16(%arg0: bf16) -> bf16 {
  %0 = math.absf %arg0 : bf16
  %1 = math.sin %0 : bf16
  return %1 : bf16
}

// CHECK-LABEL: @sequences_f16
// CHECK-SAME: ([[ARG0:%.+]]: f16)
// CHECK: [[ABSF:%.+]] = math.absf [[ARG0]]
// CHECK: [[EXTF1:%.+]] = arith.extf [[ABSF]]
// CHECK: [[SIN:%.+]] = math.sin [[EXTF1]]
// CHECK: [[TRUNCF1:%.+]] = arith.truncf [[SIN]]
// CHECK: return [[TRUNCF1]] : f16
func.func @sequences_f16(%arg0: f16) -> f16 {
  %0 = math.absf %arg0 : f16
  %1 = math.sin %0 : f16
  return %1 : f16
}
