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

# Runs the GC kernel benchmark of the e2e performance cases, python/test/e2e/bench.py.
# Logs go to stderr; stdout is a markdown table of the kernel device times, ms.
# Usage: kernel-bench.sh [-o REPORT] [bench.py options], -o saves the
# bench-report.py report. The graph_compiler package must be importable or
# located in GC_PYTHONPATH, $PROJECT_DIR/python or $GC_BUILD_DIR/python.

: ${PROJECT_DIR:="$(cd "$(dirname "$0")/.." && pwd)"}
. "$PROJECT_DIR/scripts/utils.sh"

bench_kernels() (
    report=
    if [ "${1:-}" = -o ]; then report=${2:?"-o requires a file"}; shift 2; fi
    if [ -z "${GC_PYTHONPATH:-}" ]; then
        for dir in "$PROJECT_DIR/python" ${GC_BUILD_DIR:+"$GC_BUILD_DIR/python"}; do
            [ ! -f "$dir/graph_compiler/__init__.py" ] || { GC_PYTHONPATH=$dir; break; }
        done
    fi
    export PYTHONPATH="$PROJECT_DIR/python/test${GC_PYTHONPATH:+:$GC_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
    res=$(mktemp)
    trap 'rm -f "$res"' EXIT
    rc=0
    "$PYTHON" -m e2e.bench --json "$res" "$@" >&2 || rc=$?
    # Report the completed cases even if some failed.
    [ -s "$res" ] || exit $rc
    "$PYTHON" "$PROJECT_DIR/scripts/bench-report.py" --kernels "$res" ${report:+--save "$report"}
    exit $rc
)

[ "$(basename "$0")" != "kernel-bench.sh" ] || bench_kernels "$@"
