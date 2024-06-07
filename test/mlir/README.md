# GC Tests

Tests structured the way MLIR part is similar to what one finds in the upstream (`mlir/unittest` for C++ tests and `mlir/test` for lits). These can be invoked by `gc-check` cmake target.

Backend specific tests are separate, gtest-based.
