#!/bin/bash -e

repo=intel/graph-compiler

# Uncomment for script debug
# set -x

print_usage() {
    echo "Usage:"
    echo "$0 "
    echo "    [ -d | --dev  ] Dev build, build LLVM in current env and place all to 'external' dir"
    echo "    [ -l | --dyn  ] Dynamical linking, requires rebuild of LLVM, activates 'dev' option"
    echo "    [ -h | --help ] Print this message"
}

DEV_BUILD=false
DYN_LINK=false
for arg in "$@"; do
    case $arg in
        -d | --dev)   
            DEV_BUILD=true 
            ;;
        -l | --dyn)
            DEV_BUILD=true 
            DYN_LINK=true 
            ;;
        -h | --help)
            print_usage
            exit 0
            ;;
        # -- means the end of the arguments; drop this, and break out of the while loop
        *) 
            echo Unsupported option: $arg
            print_usage
            exit 1
            ;;
    esac
done

cd $(dirname "$0")/..
PROJECT_DIR=$PWD
LLVM_HASH=$(cat cmake/llvm-version.txt)

load_llvm() {
    local run_id

    run_id=$(gh run list -w "LLVM Build" --repo $repo --json databaseId --jq '.[0].databaseId')

    gh run download "$run_id" \
        --repo "$repo" \
        --pattern "llvm-$LLVM_HASH" \
        --dir "$llvm_dir"
    cd "$llvm_dir"
    tar -zxf "llvm-$LLVM_HASH"/llvm.tgz

    MLIR_DIR="$PWD/lib/cmake/mlir"
    cd "$PROJECT_DIR"
}

build_llvm() {
    if ! [ -d "llvm-project" ]; then
        git clone https://github.com/llvm/llvm-project.git
    fi

    cd llvm-project
    git checkout ${LLVM_HASH}

    dylib=OFF
    if [ "$DYN_LINK" = 'true' ]; then 
        dylib=ON
    fi

    cmake -G Ninja llvm -B build \
        -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=true \
        -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="X86" \
        -DLLVM_INSTALL_UTILS=true -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DLLVM_INSTALL_GTEST=ON -DLLVM_BUILD_LLVM_DYLIB=$dylib -DLLVM_LINK_LLVM_DYLIB=$dylib
    cmake --build build 

    MLIR_DIR="$PWD/build/lib/cmake/mlir"
    cd ..
}

# MLIR_DIR is set on all passes
get_llvm() {
    if [ "$DEV_BUILD" = 'true' ]; then
        mkdir -p externals
        cd externals
        build_llvm 
        cd ..
        return 0
    fi

    llvm_dir=$PROJECT_DIR/../install/llvm
    if ! [ -f "$llvm_dir/llvm-$LLVM_HASH"/llvm.tgz ]; then
        load_llvm 
    else 
        MLIR_DIR="$llvm_dir/lib/cmake/mlir"
    fi
    return 0
}

get_llvm

FETCH_DIR=$PROJECT_DIR/build/_deps
DYLIB=OFF
# written in this form to set LIT_PATH in any case
if ! LIT_PATH=$(which lit) ; then
    if [ "$DEV_BUILD" != 'true' ]; then
        echo "========Warning======="
        echo "   Lit not found.     "
        echo "======================"
    fi
fi
if [ "$DEV_BUILD" = 'true' ]; then
    FETCH_DIR=$PROJECT_DIR/externals
    LIT_PATH=$PROJECT_DIR/externals/llvm-project/build/bin/llvm-lit
fi
if [ "$DYN_LINK" = 'true' ]; then 
    DYLIB=ON
fi

cmake -S . -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMLIR_DIR=$MLIR_DIR \
    -DLLVM_EXTERNAL_LIT=$LIT_PATH \
    -DFETCHCONTENT_BASE_DIR=$FETCH_DIR \
    -DGC_DEV_LINK_LLVM_DYLIB=$DYLIB

cmake --build build --parallel $(nproc)
