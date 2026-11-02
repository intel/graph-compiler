#!/bin/sh
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
. "$PROJECT_DIR/scripts/ov-build.sh" >&2

# Logs go to stderr; stdout is only a markdown table of mean latencies, ms.
bench_ov() (
    bin_dir="$OV_INSTALL_DIR/bin/intel64/$OV_BUILD_TYPE"
    bench="$OV_INSTALL_DIR/bench/sd3.5.py"
    res=$(mktemp)
    trap 'rm -f "$res"' EXIT
    unset OV_GPU_MLIR_PATTERNS OV_GPU_ENABLE_MLIR
    export LD_LIBRARY_PATH="$bin_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="$bin_dir/python${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

    run() {
        name=$1
        shift
        echo "$name${OV_GPU_MLIR_PATTERNS:+ OV_GPU_MLIR_PATTERNS=$OV_GPU_MLIR_PATTERNS}" >&2
        : >"$res"
        python3 "$bench" --profile --warmup 10 --iters 50 "$@" --result "$res" >&2
        python3 -c 'import json, sys; print("| %s | %.3f |" % (sys.argv[1], json.load(open(sys.argv[2]))["mean_ms"]))' "$name" "$res"
    }

    printf '| Benchmark | Mean |\n|---|--:|\n'
    run 'SD3.5 dynamic (no MLIR)'
    run 'SD3.5 static (no MLIR)' --static
    export OV_GPU_ENABLE_MLIR=1
    run 'SD3.5 dynamic'
    # run 'SD3.5 static' --static
    export OV_GPU_MLIR_PATTERNS='sdpa=ScaledDotProductAttention;rms1=MatMul,Add,Reshape,Transpose,RMS,MatMul,Add,Reshape,Transpose,RMS,Concat;rms2=MatMul,Add,Reshape,Transpose,MatMul,Add,Reshape,Transpose,Concat'
    # run 'SD3.5 static (patterns)' --static
)

[ "$(basename "$0")" != "ov-bench.sh" ] || bench_ov
