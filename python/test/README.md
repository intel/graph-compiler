# Graph Compiler Testing

Graph Compiler uses complementary test layers:

- Compiler IR tests in `test/mlir` check transformations and expected IR with lit
  and FileCheck.
- Direct-MLIR Python tests in this directory compile explicit MLIR, execute the
  generated GPU program, and compare numerical results with a NumPy reference.
- PyTorch integration tests in `pytorch/` exercise the Python integration,
  including model export through torch-mlir where supported.

The shared direct-MLIR infrastructure is operation-independent. Each test supplies
its MLIR, inputs, reference computation, and comparison tolerances.

```text
python/test/
  conftest.py                 # cache policy and debug options
  pytorch/
    test_torch.py            # existing PyTorch integration tests
  e2e/
    conftest.py              # configured runner fixture
    runner.py                # standalone numerical execution and benchmark helpers
    bench.py                 # kernel benchmark of the performance cases
    perf/                    # performance cases: MLIR, inputs, reference
      grouped_matmul.py
      sdpa.py
    reference/
      grouped_matmul.py      # independent NumPy references
      sdpa.py
    tests/
      test_grouped_matmul.py
      test_perf_cases.py     # correctness of the performance cases
```

## Requirements

GPU execution requires Graph Compiler built against the revision in
`cmake/llvm-version.txt`, with the repository's required LLVM patches,
`GC_ENABLE_ZE=ON`, and `GC_ENABLE_PYTHON_BINDINGS=ON`. A supported Intel GPU and
its Level Zero driver must be available.

The `python[test-e2e]` extra provides NumPy and pytest for direct-MLIR tests without
requiring PyTorch or torch-mlir. PyTorch integration tests need the `python[test]`
extra and its compatible dependencies.

The built `graph_compiler` package must be importable by the test interpreter.
Install it in the active environment or add its parent directory to `PYTHONPATH`.

## Running Tests

Run commands from the repository root. To run the direct-MLIR GPU tests without
importing the optional PyTorch suite:

```sh
python -m pytest -v python/test/e2e -m gpu
```

Select an individual test module or use pytest's `-k` option to narrow the cases.
For example:

```sh
python -m pytest -v python/test/e2e/tests/test_grouped_matmul.py
```

Run the separate PyTorch integration suite when its dependencies are installed:

```sh
python -m pytest -v python/test/pytorch
```

The shared autouse fixture sets `GC_FILE_CACHE_ON=0` for every Python test.
Tests that specifically exercise caching can request the `enable_cache` fixture;
pytest restores the environment after each test.

Use `--gc-dump-input` to print the substituted input MLIR and `--gc-dump` to
enable `GpuModule` pipeline dumps. Add `-s` to see captured output for passing
tests, for example:

```sh
python -m pytest -v -s python/test/e2e -k balanced --gc-dump-input --gc-dump
```

The dump options apply to tests using the `run_e2e` fixture. The binding currently
exposes only the `dump` compilation option; other pipeline flags need binding
support before they can be forwarded here.

## Shared Execution Flow

The `run_e2e` helper in `e2e/runner.py` implements the common numerical testing flow:

1. Copy the supplied inputs into contiguous NumPy arrays.
2. Compute expected outputs on the CPU with the supplied reference function.
3. Allocate output arrays matching the reference shapes and dtypes. Floating-point
   outputs start with NaNs so unwritten elements fail comparison with finite results.
4. Compile the supplied MLIR through the GC GPU pipeline using `GpuModule`.
5. Copy arrays to Level Zero USM, execute the program, wait for completion, and copy
   results back through the existing Python bindings.
6. Compare outputs with `numpy.testing.assert_allclose` using explicit relative
   and absolute tolerances.

Kernel arguments are ordered as inputs followed by outputs. A reference function
returns one NumPy array or a tuple of arrays for multiple outputs. The MLIR entry
point must use the corresponding argument order, shapes, and element types.

## Adding Tests

Create an operation-specific `test_*.py` module under `e2e/tests` that supplies:

- One MLIR module or parameterized template for the operation or fused graph.
- Deterministic NumPy inputs and an independent NumPy reference function.
- Explicit tolerances appropriate for the operation and accumulation precision.
- Parameterized cases for relevant shapes, supported dtypes, and boundary conditions.

Put the reference implementation in `e2e/reference`. Use `string.Template` for
parameterized MLIR so literal MLIR braces do not need escaping.
Request the `run_e2e` fixture and call
`run_e2e(mlir, inputs, reference, rtol=..., atol=...)` instead of duplicating
compilation, memory management, execution, or comparison code. Mark GPU-dependent
tests with `pytest.mark.gpu`. Execute the compiled operation on the GPU and use
the CPU reference only to compute expected results.
The current runner is exercised with floating-point kernel outputs.

`test_grouped_matmul.py` is an example: it reuses one MLIR template and reference
across balanced, uneven, and empty expert groups. Additional cases extend the
parameter table rather than duplicating the test flow.

## Kernel Benchmarks

The performance cases in `e2e/perf` are small MLIR fragments, for example, the
SDPA pattern that the OpenVINO GPU plugin offloads to GC in SD3.5. The model
benchmark only shows the total latency, where a regression of a single kernel by
a few percent is lost in the noise, so these fragments are benchmarked separately.

`python -m e2e.bench` compiles each case through the GC GPU pipeline, checks the
results against the reference, then runs the program `--warmup` (50) and
`--iters` (200) times on the same device buffers. The device time of a run is
the sum of its kernels' execution times, measured with the Level Zero kernel
timestamps (`graph_compiler.set_profiling()` and
`graph_compiler.take_kernel_times()`), so it excludes the launch overhead and the
copies. It is the same metric as `IMEX_ENABLE_PROFILING` of the IMEX L0 runtime
patch. The median is the most stable statistic and is compared in CI.

```sh
cd python/test
python -m e2e.bench --list
python -m e2e.bench -k sdpa --iters 500 --json /tmp/sdpa.json
```

`scripts/kernel-bench.sh` runs it and prints a markdown table,
`scripts/bench-report.py` merges it with `scripts/ov-bench.sh` results and
compares them with a baseline report:

```sh
scripts/kernel-bench.sh -o new.json
scripts/bench-report.py --input new.json --baseline old.json
```

To add a case, append a `Case` to the `CASES` list of a module in `e2e/perf`,
see `e2e/perf/__init__.py`. The `Source MLIR` fragment, printed by the OpenVINO
GPU plugin with `OV_GPU_MLIR_DEBUG=1`, can be used as is: the kernel arguments are the
inputs followed by the outputs. `tests/test_perf_cases.py` checks the results of
every case, so a case must have a reference.

## Failures and Skips

Every direct-MLIR test executes on the GPU; CPU computation is used only to
produce reference outputs for comparison. Missing compiled bindings are failures.
Compilation, runtime, and numerical errors are failures, not skips. Do not mask a
failure by broadening tolerances without investigating the numerical difference.