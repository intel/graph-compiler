#! /bin/bash

# Copyright (C) 2024 Intel Corporation
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# 
# SPDX-License-Identifier: Apache-2.0


# this is for developer local check only.
# not used by github action

check_tool() {
  which $1 &> /dev/null || (echo "$1 not found!" && exit 1)
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --format)
      CHECK_FORMAT=1
      shift
      ;;
    --tidy)
      CHECK_TIDY=1
      shift
      ;;
    --license)
      CHECK_LICENSE=1
      shift
      ;;
    *) # Handle other unknown options
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# set PR_REF if your pr target is not main
: ${PR_REF:=main}


PROJECT_ROOT=$(pwd)/$(dirname "${BASH_SOURCE[0]}")/../
cd $PROJECT_ROOT

MERGE_BASE=$(git merge-base $PR_REF HEAD)
CHANGED_FILES=$(git diff --name-only $MERGE_BASE HEAD)

# if you do not have clang/clang++/clang-tidy/clang-format
# please install it
# conda install -c conda-forge clangxx cxx-compiler clang-tools

if [ -n "$CHECK_TIDY" ]; then
  echo "start tidy check..."
  if [ -z "${MLIR_DIR}" ]; then
    echo "The environment variable MLIR_DIR is not set."
    exit 1
  fi
  check_tool "clang"
  check_tool "clang++"
  check_tool "clang-tidy"
  check_tool "lit"

  TIDY_ROOT=${PROJECT_ROOT}/build/tidy
  mkdir -p ${TIDY_ROOT}
  cd ${TIDY_ROOT}
  cmake ../../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR=${MLIR_DIR} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=True \
    -DCMAKE_C_COMPILER=$(which clang) \
    -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DLLVM_EXTERNAL_LIT=$(which lit)

  for f in $(find ./include -name Makefile); do
    targets=$(make -f $f help |grep IncGen); 
    if [[ $? -eq 0 ]]; then 
      for target in $targets; do
        cd ${f%Makefile} && make ${target#...} && cd -; 
      done
    fi ;
  done

  [ -f run-clang-tidy.py ] || wget https://raw.githubusercontent.com/llvm/llvm-project/main/clang-tools-extra/clang-tidy/tool/run-clang-tidy.py -O run-clang-tidy.py
  [ -f .clang-tidy ] || wget https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/.clang-tidy -O .clang-tidy
  
  python3 run-clang-tidy.py -warnings-as-errors=* -p ./ -config-file .clang-tidy -clang-tidy-binary $(which clang-tidy) $CHANGED_FILES
fi

if [ -n "$CHECK_FORMAT" ]; then
  echo "start format check..."
  check_tool "clang-format"
  cd $PROJECT_ROOT
  echo "$CHANGED_FILES" | egrep "*\\.(h|hpp|c|cpp)$" | xargs clang-format --dry-run --Werror -style=file
fi

if [ -n "$CHECK_LICENSE" ]; then
  echo "start license check..."
  cd $PROJECT_ROOT
  python3 scripts/license.py --files $(echo $CHANGED_FILES | tr ' ' ',')
fi