################################################################################
# Copyright (C) 2026 Intel Corporation
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
# SPDX-License-Identifier: Apache-2.0
################################################################################

: ${PROJECT_DIR:="$(cd "$(dirname "$0")/.." && pwd)"}
. "$PROJECT_DIR/scripts/ov-build.sh"

# Regex patterns of tests to exclude from the run.
: ${OV_TEST_EXCLUDE:=""}
: ${OV_TEST_NPROC:=32}
: ${OV_TEST_DUMP_DIR:="$PWD/failed_test_dumps/ov"}
test_ov() {
    export OV_GPU_ENABLE_MLIR=1
    export OV_GPU_QUEUE_TYPE=out-of-order
    # Without this, mlir_RMS test fails due to shape [1,64,1,1].
    export OV_GPU_ALLOW_NEW_SHAPE_INFER=1
    ov_test_runner="$OV_INSTALL_DIR/bin/intel64/Release/ov_gpu_func_tests"
    test_runner="$PROJECT_DIR/scripts/run-test.sh"
    rm -rf "$OV_TEST_DUMP_DIR"

    filter=$(printf '%s:' \
        'MLIRExecution.SimpleMatmulf16' \
        'MLIRExecution.SDPABasic' \
        '*ScaledAttnLayerGPUMlirTest*' \
        'mlir_*' \
    )

    export LD_LIBRARY_PATH="$OV_INSTALL_DIR/bin/intel64/Release:$LD_LIBRARY_PATH"
    tests=$("$ov_test_runner" --gtest_list_tests --gtest_filter="${filter%:}" \
        | awk -v exclude="$OV_TEST_EXCLUDE" '/^  /{print suite $1} /^[^ ]/{suite=$1}')
    [ -n "$OV_TEST_EXCLUDE" ] && tests=$(echo "$tests" | grep -v -E "$OV_TEST_EXCLUDE")
    [ -n "$tests" ] || { echo "No tests found" >&2 && exit 1; }

    start=$(date +%s)
    echo "$tests" | xargs -P "$OV_TEST_NPROC" -I{} "$test_runner" "$OV_TEST_DUMP_DIR" "$ov_test_runner" --gtest_filter='{}'
    echo "Run $(echo "$tests" | wc -l) OV tests in $(($(date +%s) - start))s"
}

[ "$(basename "$0")" != "ov-test.sh" ] || test_ov
