# Graph Compiler
Graph Compiler is in active development stage.

## Build instructions

All-on-one compile script is at `scripts/compile.sh`.

To build this project step by step, first you need to find the LLVM commit-id we are using at `cmake/llvm-version.txt`. Then clone specific version of LLVM:

```bash
LLVM_COMMIT=????? # the commit id in cmake/llvm-version.txt of this repo
git clone https://github.com/llvm/llvm-project
cd llvm-project
git checkout $LLVM_COMMIT
```

Build LLVM with the command lines given in `.github/workflows/build-llvm.yml`:

```bash
mkdir llvm-install
cmake -G Ninja llvm -B build -DCMAKE_INSTALL_PREFIX=llvm-install \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=true -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_INSTALL_UTILS=true -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLVM_INSTALL_GTEST=ON
cmake --build build --target install
```

We have now installed LLVM at `llvm-project/llvm-install`.

Clone the graph-compiler repo:

```bash
cd ../.. # working directory changed to parent directory of llvm-project
git clone https://github.com/intel/graph-compiler.git
cd graph-compiler
mkdir build && cd build
```

Build and run tests:

```bash
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMLIR_DIR=/PATH/TO/llvm-project/llvm-install/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=$(which lit)
cmake --build . --target gc-check
```

Notes:
 * `/PATH/TO/llvm-project/llvm-install` should be the install path of LLVM. If you installed LLVM elsewhere by `-DCMAKE_INSTALL_PREFIX` option when building LLVM, you need to change the path in `-DMLIR_DIR` accordingly.
 *  The cmake option `-DLLVM_EXTERNAL_LIT` is for the tests of this project. It requires the `lit` tool to be installed in the system. You can install it via `pip install llvm-lit`. If you don't need to run the tests of this repo, you can omit this option in the command line.

Optional cmake options:
 * `-DGC_LEGACY_ENABLE=ON/OFF` turn on/off the legacy graph-compiler component. By default `ON`.
 * `-DGC_TEST_ENABLE=ON/OFF` turn on/off building the tests . By default `ON`.
 * `-DGC_DEV_LINK_LLVM_DYLIB=ON/OFF` link the dynamic LLVM/MLIR libraries if available. This option is for developer's use. By default `OFF`.
