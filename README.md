## Build instructions

### Build the project

To build the project, simply run:

```bash
./scripts/compile.sh
```

### Build and install Python bindings

Create a virtual environment and install dependencies:

```bash
python -m venv .venv --prompt gc
source .venv/bin/activate
pip install nanobind torch --index-url https://download.pytorch.org/whl/xpu
```

Install the Python bindings in editable mode:

```bash
pip install -e python/
```

### Run Python examples

```bash
python python/example/matmul.py
```

### Run gc-opt

```bash
./build/bin/gc-opt --gpu-dev-props='arch=pvc' --gc-gpu-pipeline='dump' linalg_on_tensors.mlir
```
