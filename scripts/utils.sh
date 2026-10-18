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
[ "${_GC_UTILS_SH_ONCE:-}" ] && return || _GC_UTILS_SH_ONCE=1
set -e

# Set a GitHub Actions environment variable if the value is not empty.
set_gh_env_var() {
  [ "$#" -eq 2 ] || return 0
  [ -n "$2" ] || return 0
  printf '%s=%s\n' "$1" "$2" >> "$GITHUB_ENV"
  export "$1=$2"
}

# Check if value is true (either "true" or "1")
is_true() {
  [ "$1" = "true" ] || [ "$1" = "1" ]
}

# Resolve the SHA of a given Git tag or branch name.
# Usage: git_tag_sha <repository> <tag_or_branch_or_sha>
git_sha() {
  echo "$2" | grep -qE '^[0-9a-f]{40}$' && { echo "$2"; return; } # If it's already a SHA
  git ls-remote "$1" "refs/heads/$2" "refs/tags/$2" | awk '{print $1}' | head -1
}

# Compute the SHA1 hash of the given files, dirs and/or strings.
# Usage: sha1 <file_or_dir_or_string>...
sha1() {
  {
    for i in "$@"; do
      [ ! -e "$i" ] || find "$i" -type f
    done | sort -u | while read i; do sha1sum "$i"; done
    for i in "$@"; do [ -e "$i" ] || echo "$i"; done
  } | sha1sum | cut -d' ' -f1;
}
