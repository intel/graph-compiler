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
