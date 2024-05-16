#!/bin/bash -e

repo=intel/graph-compiler

# Uncomment for script debug
# set -ex

function print_usage() {
    echo "Usage:"
    echo "$0 "
    echo "    [ -d | --dev  ] Dev build, build LLVM in current env and place all to 'external' dir"
    echo "    [ -l | --dyn  ] Dynamical linking, requires rebuild of LLVM, activates 'dev' option"
    echo "    [ -h | --help ] Print this message"
}

args=$(getopt -a -o dlh --long dev,dyn,help -- "$@")
if [[ $? -gt 0 ]]; then
    echo "first"
    print_usage
fi

DEV_BUILD=false
DYN_LINK=false
eval set -- ${args}
while :
do
    case $1 in
        -d | --dev)   
            DEV_BUILD=true 
            shift   ;;
        -l | --dyn)
            DEV_BUILD=true 
            DYN_LINK=true 
            shift   ;;
        -h | --help)
            echo "in help"    
            print_usage
            shift   ;;
        # -- means the end of the arguments; drop this, and break out of the while loop
        --) shift; break ;;
        *) >&2 echo Unsupported option: $1
            echo "in unsup"
            print_usage ;;
    esac
done

cd $(dirname "$0")/..
project_dir=$PWD
llvm_hash=$(cat cmake/llvm-version.txt)

function load_llvm() {
    local  mlir_dir=$1
    local run_id

    run_id=$(gh run list -w "LLVM Build" --repo $repo --json databaseId --jq '.[0].databaseId')

    gh run download "$run_id" \
        --repo "$repo" \
        --pattern "llvm-$llvm_hash" \
        --dir "$llvm_dir"
    cd "$llvm_dir"
    tar -zxf "llvm-$llvm_hash"/llvm.tgz

    eval $mlir_dir="$PWD/lib/cmake/mlir"
    cd "$project_dir"
    return 0
}

function build_llvm() {
    local  mlir_dir=$1

    if ! [ -d "llvm-project" ]; then
        git clone https://github.com/llvm/llvm-project.git
    fi

    cd llvm-project
    git checkout ${llvm_hash}

    dylib=OFF
    if [[ "$DYN_LINK" == 'true' ]]; then 
        dylib=ON
    fi

    cmake -G Ninja llvm -B build \
        -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=true \
        -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="X86" \
        -DLLVM_INSTALL_UTILS=true -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DLLVM_INSTALL_GTEST=ON -DLLVM_BUILD_LLVM_DYLIB=$dylib -DLLVM_LINK_LLVM_DYLIB=$dylib
    cmake --build build 

    eval $mlir_dir="$PWD/build/lib/cmake/mlir"
    cd ..
    return 0
}

function get_llvm() {
    local ret_val=$1

    if [[ "$DEV_BUILD" == 'true' ]]; then
        mkdir -p externals
        cd externals
        build_llvm val
        eval $ret_val=\$val
        cd ..
        return 0
    fi

    llvm_dir=$project_dir/../install/llvm
    if ! [ -f "$llvm_dir/llvm-$llvm_hash"/llvm.tgz ]; then
        load_llvm val
        eval $ret_val=\$val
    fi
    eval $ret_val="$llvm_dir/lib/cmake/mlir"
    return 0
}

get_llvm mlir_dir

fetch_dir=$project_dir/build/_deps
dylib=OFF
lit_path=""
if [[ $(which lit) ]]; then
    lit_path=$(which lit)
fi
if [[ "$DEV_BUILD" == 'true' ]]; then
    fetch_dir=$project_dir/externals
    lit_path=$project_dir/externals/llvm-project/build/bin/llvm-lit
fi
if [[ "$DYN_LINK" == 'true' ]]; then 
    dylib=ON
fi
if [ -z "$lit_path" ]; then 
    echo "========Warning======="
    echo "   Lit not found.     "
fi

cmake -S . -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMLIR_DIR=$mlir_dir \
    -DLLVM_EXTERNAL_LIT=$lit_path \
    -DFETCHCONTENT_BASE_DIR=$fetch_dir \
    -DGC_DEV_LINK_LLVM_DYLIB=$dylib
cmake --build build --parallel $(nproc)
