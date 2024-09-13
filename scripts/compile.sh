#!/bin/bash -e

cd "$(dirname "$0")/.."
PROJECT_DIR="$PWD"
EXTERNALS_DIR="$PWD/externals"
REPO=intel/graph-compiler

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
    [ -i | --imex    ] Compile with IMEX (used for GPU pipeline)
    [ -h | --help    ] Print this message
EOF
}

DYN_LINK=OFF
ENABLE_IMEX=false
for arg in "$@"; do
    case $arg in
        -d | --dev)   
            DEV_BUILD=true 
            ;;
        -i | --imex)
            ENABLE_IMEX=true
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

if [ "$ENABLE_IMEX" = "true" ] && [ "$DYN_LINK" = "ON" ]; then
    echo "IMEX doesn't support dynamical linking of LLVM"
    exit 1
fi

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

if [ "$ENABLE_IMEX" = "true" ]; then
    LLVM_HASH=$(cat cmake/llvm-version-imex.txt)
else
    LLVM_HASH=$(cat cmake/llvm-version.txt)
fi

load_llvm() {
    if [ "$ENABLE_IMEX"  = "true" ]; then
        local llvm_version="llvm-${LLVM_HASH}-imex-patched"
    else
        local llvm_version="llvm-${LLVM_HASH}"
    fi

    gh run download \
        --repo "$REPO" \
        -n "$llvm_version" \
        --dir "$llvm_dir"
    tar -zxf llvm.tgz

    MLIR_DIR="$PWD/lib/cmake/mlir"
}

build_llvm() {
    local llvm_dir="$EXTERNALS_DIR/llvm-project"
    if ! [ -d "$llvm_dir" ]; then
        git clone https://github.com/llvm/llvm-project.git
        cd "$llvm_dir"
    else
        cd "$llvm_dir"
        git fetch --all
        # discard all unstaged changes (there could be remaining patches from the IMEX
        # build that would break 'git checkout ${LLVM_HASH}')
        git checkout -- .
    fi

    git checkout ${LLVM_HASH}
    [ -z "$CLEANUP" ] || rm -rf build

    [ "$DYN_LINK" = "OFF" ] && CXX_FLAGS="-fvisibility=hidden"

    if [ "$ENABLE_IMEX" = "true" ]; then
        # clone IMEX and apply patches
        local mlir_ext_dir="$EXTERNALS_DIR/mlir-extensions"
        if ! [ -d "$mlir_ext_dir" ]; then
            cd "$EXTERNALS_DIR"
            git clone https://github.com/intel/mlir-extensions.git
            cd "$mlir_ext_dir"
        else
            cd "$mlir_ext_dir"
            git fetch --all
        fi

        IMEX_HASH=$(cat "$PROJECT_DIR/cmake/imex-version.txt")
        git checkout ${IMEX_HASH}

        cd "$llvm_dir"
        find "$mlir_ext_dir/build_tools/patches" -name '*.patch' | sort -V | xargs git apply
    fi

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
        -DLLVM_INCLUDE_TESTS=ON \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_INCLUDE_DOCS=OFF \
        -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_INSTALL_GTEST=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    cmake --build build --parallel $MAX_JOBS

    MLIR_DIR="$PWD/build/lib/cmake/mlir"
}

# MLIR_DIR is set on all passes
get_llvm() {
    if [ ! -z "$DEV_BUILD" ]; then
        mkdir -p "$EXTERNALS_DIR"
        cd "$EXTERNALS_DIR"
        build_llvm
        cd "$PROJECT_DIR"
        return 0
    fi

    local llvm_dir="$PROJECT_DIR/../install/llvm"
    if ! [ -f "$llvm_dir/$llvm_name/llvm.tgz" ]; then
        mkdir -p "$llvm_dir"
        cd "$llvm_dir"
        load_llvm
        cd "$PROJECT_DIR"
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
    -DGC_DEV_LINK_LLVM_DYLIB=$DYN_LINK \
    -DGC_ENABLE_IMEX=$ENABLE_IMEX

cmake --build build --parallel $MAX_JOBS
