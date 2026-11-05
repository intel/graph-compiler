## Build and install Python bindings

Create a virtual environment:

```bash
./scripts/create-venv.sh
. .venv/bin/activate
```

Build the project and install the Python bindings in editable mode.
`--no-build-isolation` is required to build against the nanobind version
installed by `create-venv.sh`, which matches the LLVM requirement:

```bash
pip install --no-build-isolation -v -e python/
```

## Run Python examples

```bash
python python/example/matmul.py
```

## Run Python unit tests

```bash
pytest python/test/test_torch.py -v -s
```

## Run gc-opt

```bash
./build/bin/gc-opt --gpu-dev-props='arch=bmg' --gc-gpu-pipeline='dump' linalg_on_tensors.mlir
```

## Portable GPU backends

Set `-DGC_ENABLE_IGC=OFF` to use generic GPU-to-SPIR-V lowering without the
XeGPU/XeVM pipeline. `GpuCompiler`, the host JIT, and `gc-opt` remain available.
IGC defaults to OFF on macOS. To select SPIR-V in an IGC-enabled build, set
`CompileOptions::spirv` or use `--gc-gpu-pipeline=spirv`. This path supports
static shapes and requires a runtime that accepts OpenCL SPIR-V. The Level Zero
runtime recognizes SPIR-V binaries automatically. Set `GC_ENABLE_ZE=OFF` if
Level Zero is unavailable.

`GC_ENABLE_OCL=ON` also builds the OpenCL C source backend. Use
`mlir::gc::gpu::OclSourceModule` from `gc/Runtime/OclSourceModule.h` to compile
and invoke kernels with OpenCL buffers, including on macOS. Attention
materializes its score matrix. Python bindings require Level Zero.

Run `ctest --test-dir <build> --output-on-failure` to check SPIR-V compilation
and JIT dispatch, plus source execution on an available OpenCL GPU.
