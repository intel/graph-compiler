## Build and install Python bindings

Create a virtual environment:

```bash
python -m venv .venv --prompt gc
source .venv/bin/activate
```

Build the project and install the Python bindings in editable mode:

```bash
pip install -e python/[test]
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
./build/bin/gc-opt --gpu-dev-props='arch=pvc' --gc-gpu-pipeline='dump' linalg_on_tensors.mlir
```
