# MLIR Binding of Graph Complier
## Pre-requisites
Building LLVM project:
1. Installation of python dependencies as specified in mlir/python/requirements.txt
2. CMake variables 
```
MLIR_ENABLE_BINDINGS_PYTHON:BOOL
Python3_EXECUTABLE:STRING
```

## Enable the support in VSCode,
1. Install the Python extension for VSCode.
2. Add the following Python stub paths to the configuration file in your workspace directory: `.vscode/settings.json`.

    ```json
    {
        "python.autoComplete.extraPaths": [
            "${workspaceFolder}/build/python_packages/gc_mlir_core/"
        ],
        "python.analysis.extraPaths": [
            "${workspaceFolder}/build/python_packages/gc_mlir_core/"
        ]
    }
    ```
The JSON config above assumes that your cmake build directory is under `${workspaceFolder}/build`. If your cmake build path is not here, please change the above path accordingly