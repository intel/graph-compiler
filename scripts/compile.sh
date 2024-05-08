#!/bin/bash -e

repo=intel/graph-compiler

cd $(dirname "$0")/..
llvm_dir=$(cd ..; pwd -P)/install/llvm
llvm_hash=$(cat cmake/llvm-version.txt)

get_llvm() (
    local run_id

    run_id=$(gh run list -w "LLVM Build" --repo $repo --json databaseId --jq '.[0].databaseId')

    gh run download "$run_id" \
       --repo "$repo" \
       --pattern "llvm-$llvm_hash" \
       --dir "$llvm_dir"
    cd "$llvm_dir"
    tar -zxf "llvm-$llvm_hash"/llvm.tgz
)

test -f "$llvm_dir/llvm-$llvm_hash"/llvm.tgz || get_llvm

cmake -S . -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMLIR_DIR=$llvm_dir/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=$(which lit)
cmake --build build --parallel $(nproc)
