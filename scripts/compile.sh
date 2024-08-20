#!/bin/bash -e

repo=intel/graph-compiler

set -e

# Uncomment for script debug
# set -x

print_usage() {
    cat <<EOF
Usage:
$(basename "$0")
    [ -d | --dev     ] Dev build, build LLVM in current env and place all to 'external' dir
    [ -l | --dyn     ] Dynamical linking, requires rebuild of LLVM, activates 'dev' option
    [ -r | --release ] Release build, requires rebuild of LLVM in Release mode, activates 'dev' option
    [ -c | --clean   ] Delete the build artifacts from the previous build
    [ -h | --help    ] Print this message
EOF
}

DYN_LINK=OFF
for arg in "$@"; do
    case $arg in
        -d | --dev)   
            DEV_BUILD=true 
            ;;
        -l | --dyn)
            DYN_LINK=ON
            DEV_BUILD=true
            ;;
        -r | --release)
            REL_BUILD=true
            DEV_BUILD=true
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

if [ ! -z "$REL_BUILD" ]; then
    BUILD_TYPE=Release
elif [ ! -z "$DEV_BUILD" ]; then
    BUILD_TYPE=Debug
else
    BUILD_TYPE=RelWithDebInfo
fi

if [ -z "$MAX_JOBS" ]; then
    MAX_JOBS=$(($(nproc) - 2))  # Do not use all CPUs
    [ $MAX_JOBS -gt 0 ] || MAX_JOBS=1
fi

cd $(dirname "$0")/..
PROJECT_DIR=$PWD
LLVM_HASH=$(cat cmake/llvm-version.txt)

load_llvm() {
    local run_id

    gh run download "$run_id" \
        --repo "$repo" \
        -n "llvm-$LLVM_HASH" \
        --dir "$llvm_dir"
    cd "$llvm_dir"
    tar -zxf llvm.tgz

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

    git checkout ${LLVM_HASH}
    # git reset --hard
    [ -z "$CLEANUP" ] || rm -rf build

    [ "$DYN_LINK" = "OFF" ] && CXX_FLAGS="-fvisibility=hidden"

    cmake -G Ninja llvm -B build \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
        -DCMAKE_CXX_FLAGS_DEBUG="-g -O0" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir"\
        -DLLVM_TARGETS_TO_BUILD="X86" \
        -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
        -DLLVM_BUILD_LLVM_DYLIB=$DYN_LINK \
        -DLLVM_LINK_LLVM_DYLIB=$DYN_LINK \
        -DLLVM_INCLUDE_RUNTIMES=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_INCLUDE_DOCS=OFF \
        -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_INSTALL_GTEST=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    cmake --build build --parallel $MAX_JOBS

    MLIR_DIR="$PWD/build/lib/cmake/mlir"
    cd ..
}

# MLIR_DIR is set on all passes
get_llvm() {
    if [ ! -z "$DEV_BUILD" ]; then
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

# written in this form to set LIT_PATH in any case
if ! LIT_PATH=$(which lit) && [ -z "$DEV_BUILD" ]; then
    echo "========Warning======="
    echo "   Lit not found.     "
    echo "======================"
fi

if [ -z "$DEV_BUILD" ]; then
    FETCH_DIR=$PROJECT_DIR/build/_deps
else
    FETCH_DIR=$PROJECT_DIR/externals
    LIT_PATH=$PROJECT_DIR/externals/llvm-project/build/bin/llvm-lit
fi

[ -z "$CLEANUP" ] || rm -rf build
cmake -S . -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DMLIR_DIR=$MLIR_DIR \
    -DLLVM_EXTERNAL_LIT=$LIT_PATH \
    -DFETCHCONTENT_BASE_DIR=$FETCH_DIR \
    -DGC_DEV_LINK_LLVM_DYLIB=$DYN_LINK

cmake --build build --parallel $MAX_JOBS
