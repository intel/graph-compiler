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
DUMP_DIR="$1"
shift

printf 'Running test: %s\n' "$*"
OUTPUT="$(printf '%s\n' "$*" && echo "Test output:" && "$@" 2>&1)"
EC=$?
if [ $EC -eq 0 ]; then
    [ "$GC_DUMP_IR" = "1" ] && printf '%s\n' "$OUTPUT"
    exit 0
fi

DUMP_FILE="$DUMP_DIR/$(printf '%s\n' "$*" | sha1sum | cut -d' ' -f1).txt"
mkdir -p "$DUMP_DIR"
printf "Test '%s' failed with exit code %s. Saving output to %s.\n" "$*" "$EC" "$DUMP_FILE"
printf '%s\n' "$OUTPUT" > "$DUMP_FILE"
[ "$GC_DUMP_IR" = "1" ] && printf '%s\n' "$OUTPUT" && exit $EC

echo "Running test again with IR dumps enabled:" >> "$DUMP_FILE"
GC_DUMP_IR=1 GC_FILE_CACHE_ON=0 OV_GPU_MLIR_DEBUG=1 "$@" >> "$DUMP_FILE" 2>&1
cat "$DUMP_FILE"
exit $EC
