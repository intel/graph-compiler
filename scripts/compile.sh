#!/bin/bash -e

repo=intel/graph-compiler

# Uncomment for script debug
# set -x

print_usage() {
    cat <<EOF
Usage:
$(basename "$0")
    [ -d | --dev   ] Dev build, build LLVM in current env and place all to 'external' dir
    [ -l | --dyn   ] Dynamical linking, requires rebuild of LLVM, activates 'dev' option
    [ -c | --clean ] Delete the build artifacts from the previous build
    [ -h | --help  ] Print this message
EOF
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
        -c | --clean)
            CLEANUP=true
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
        cd llvm-project
    else
        cd llvm-project
        git fetch --all
    fi

    git pull
    git checkout ${LLVM_HASH}

    dylib=OFF
    if [ "$DYN_LINK" = 'true' ]; then 
        dylib=ON
    fi

    python -m pip install -r mlir/python/requirements.txt

    [ -z "$CLEANUP" ] || rm -rf build
    cmake -G Ninja llvm -B build \
        -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-g -O0" \
        -DLLVM_ENABLE_ASSERTIONS=true -DLLVM_ENABLE_PROJECTS="mlir"\
        -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_INSTALL_UTILS=true \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLVM_INSTALL_GTEST=ON \
        -DLLVM_BUILD_LLVM_DYLIB=$dylib -DLLVM_LINK_LLVM_DYLIB=$dylib \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE=$(which python3)
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
    BUILD_TYPE=Debug
    FETCH_DIR=$PROJECT_DIR/externals
    LIT_PATH=$PROJECT_DIR/externals/llvm-project/build/bin/llvm-lit
else
    BUILD_TYPE=RelWithDebInfo
fi
if [ "$DYN_LINK" = 'true' ]; then 
    DYLIB=ON
fi

[ -z "$CLEANUP" ] || rm -rf build
cmake -S . -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DMLIR_DIR=$MLIR_DIR \
    -DLLVM_EXTERNAL_LIT=$LIT_PATH \
    -DFETCHCONTENT_BASE_DIR=$FETCH_DIR \
    -DGC_DEV_LINK_LLVM_DYLIB=$DYLIB \
    -DGC_LEGACY_ENABLE=OFF -DGC_TEST_ENABLE=OFF

cmake --build build --parallel $(nproc)
