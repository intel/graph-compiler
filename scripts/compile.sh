#!/bin/bash -e

repo=intel/graph-compiler

cd $(dirname "$0")/..
llvm_dir=$(cd ..; pwd -P)/install/llvm

get_llvm() (
    local run_id

    run_id=$(gh run list -w "LLVM Build" --repo $repo --json databaseId --jq '.[0].databaseId')

    gh run download "$run_id" \
       --repo "$repo" \
       --pattern "llvm-*" \
       --dir "$llvm_dir"
    cd "$llvm_dir"
    tar -zxf llvm-*/llvm.tgz
)

test -f "$llvm_dir"/llvm-*/llvm.tgz || get_llvm

cmake -S . -G Ninja -B build
cmake --build build --parallel $(nproc)
