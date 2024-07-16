# BenchGC Overview

BenchGC is a testing tool for the Graph Compiler project, currently focusing on verifying the correctness of the Graph Compiler.

### What can BenchGC do?
* Single MLIR Op validation
* User-provided MLIR Module validation
* Data filling strategy
* Result comparison strategy

### BenchGC workflow
```mermaid
flowchart TB
  TEST_OP["Dialect+Op"] --> SINGLE_OP_TEST["Single Op Test"]
  SHAPE["Shape"] --> SINGLE_OP_TEST["Single Op Test"]
  DTYPE["Data Type"] --> SINGLE_OP_TEST["Single Op Test"]
  FILLING["Filling strategy"] --> SINGLE_OP_TEST["Single Op Test"]
  COMPARE["Compare Strategy"] --> SINGLE_OP_TEST["Single Op Test"]

  FILLING["Filling strategy"] --> USER_MLIR_TEST["User MLIR Test"]
  COMPARE["Compare Strategy"] --> USER_MLIR_TEST["User MLIR Test"]  

  SINGLE_OP_TEST["Single Op Test"] --> MLIR_MODULE["MLIR Module"]
  USER_MLIR_TEST["User MLIR Test"] -- Load a MLIR file --> MLIR_MODULE["MLIR Module"]
  MLIR_MODULE["MLIR Module"] --> REF_MLIR_RUNNER["Reference MLIR Runner"]
  MLIR_MODULE["MLIR Module"] --> GRAPH_COMPILER_LIBRARY["Graph Compiler Library"]
  REF_MLIR_RUNNER["Reference MLIR Runner"] -- traverse and translate MLIR and run with pytorch --> REF_RESULT["Reference Result"]
  GRAPH_COMPILER_LIBRARY["Graph Compiler Library"] --> GC_RESULT["Graph Compiler Result"]
  REF_RESULT["Reference Result"] --> COMPARISON["Compare based on strategy"]
  GC_RESULT["Graph Compiler Result"] --> COMPARISON["Compare based on strategy"]
  COMPARISON["Compare based on strategy"] --> BENCH_RESULT["Success or Failure"]
```

