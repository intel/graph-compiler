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

import datetime, sys, re, argparse
from typing import Dict, Set

WIDTH: int = 80
intel_license: list[str] = [
    'Copyright \\(C\\) (\\d\\d\\d\\d-)?$YEAR Intel Corporation',
    '',
    'Licensed under the Apache License, Version 2.0 (the "License");',
    'you may not use this file except in compliance with the License.',
    'You may obtain a copy of the License at',
    '',
    'http://www.apache.org/licenses/LICENSE-2.0',
    '',
    'Unless required by applicable law or agreed to in writing,',
    'software distributed under the License is distributed on an "AS IS" BASIS,',
    'WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.',
    'See the License for the specific language governing permissions',
    'and limitations under the License.',
    'SPDX-License-Identifier: Apache-2.0',
]

llvm_license: list[str] = [
    "===-{1,2} $FILE - .* -*\\*- $LANG -\\*-===",
    '',
    'This file is licensed under the Apache License v2.0 with LLVM Exceptions.',
    'See https://llvm.org/LICENSE.txt for license information.',
    'SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception',
    '',
    "===-*===",
]

def check_license(filepath: str, license: list[str], var: Dict[str, str], re_line: Set[int]):
    with open(filepath, 'r') as f:
        idx: int = 0
        for line in f.readlines():
            lic: str = license[idx]
            # replace the variable defined in license
            for k, v in var.items():
                lic = lic.replace(k, v)
            if idx in re_line:
                if re.search(lic, line) is not None and ("RE_WIDTH" not in var or int(var["RE_WIDTH"]) + 1 == len(line)):
                    idx += 1
            elif line.find(lic) != -1:
                idx += 1
            if idx == len(license):
                return True
        return False

def fix_intel_license(var: Dict[str, str]):
    lang: str = var['$LANG']
    cmt: str = "" # comment char
    if lang == "C\\+\\+":
        print("/*")
        cmt = " *"
    elif lang == "cmake" or lang == "Python":
        print("#" * WIDTH)
        cmt = "#"
    print('%s Copyright (C) %s Intel Corporation' % (cmt, var['$YEAR']))   
    for i in range(1, len(intel_license)):
        print('%s %s' % (cmt, intel_license[i]))   
    if var['$LANG'] == "C\\+\\+":
        print(" */")
    elif lang == "cmake" or lang == "Python":
        print("#" * WIDTH)

def fix_llvm_license(var: Dict[str, str]):
    lang: str = var['$LANG']
    cmt: str = "//" # comment char
    if lang == "Python":
        cmt = "#"
    elif lang == "C\\+\\+":
        lang = "C++"

    part1 = "%s===-- %s - DESC " % (cmt, var['$FILE'])
    part3 = "-*- %s -*-===%s" % (lang, cmt)
    part2 = "-" * (WIDTH - len(part1) - len(part3))

    print(part1 + part2 + part3)
    for i in range(1, len(llvm_license) - 1):
        print((cmt + " " + llvm_license[i]).rstrip())
    part1 = cmt + "==="
    part3 = "===" + cmt
    part2 = "-" * (WIDTH - len(part1) - len(part3))
    print(part1 + part2 + part3)
        
def use_llvm_license(path: str) -> bool:
    for folder in ["lib/gc/", 'include/gc/', 'unittests/', 'python/gc_mlir']:
        if path.startswith(folder) or path.startswith('./' + folder):
            return True
    return False
                
year: int = datetime.datetime.now().year
success: bool = True
                
parser = argparse.ArgumentParser(prog = "benchgc license checker")
parser.add_argument("--files", required=True, type = str, help = "comma seperated file list")
args = parser.parse_args()

for filepath in args.files.split(','):
    name: str = filepath.split('/')[-1]
    var: Dict[str, str] = {}
    re_line: Set[int] = set()

    lic = list[str]

    if filepath.startswith("test/") or filepath.startswith("./test/"):
        continue

    if name.endswith(".py"):
        var['$LANG'] = "Python"
    else:
        for suffix in [".c", ".cpp", ".h", ".hpp"]:
            if name.endswith(suffix):
                var['$LANG'] = "C\\+\\+"

    is_llvm_license = use_llvm_license(filepath)
    if is_llvm_license:
        # llvm license, only check python/cpp now
        lic = llvm_license
        re_line.add(0)
        re_line.add(6)
        var['$FILE'] = name
        # the line we read contains a '\n' character, so the expected length should be 81
        var['RE_WIDTH'] = str(WIDTH)
        if name.endswith(".td"):
            var['$LANG'] = "tablegen"
    else:
        # intel license
        lic = intel_license
        re_line.add(0)
        var['$YEAR'] = str(year)
        if name == "CMakeLists.txt":
            var['$LANG'] = "cmake"

    if "$LANG" not in var:
        continue
    if not check_license(filepath, lic, var, re_line):
        success = False
        print("Fail         : %s" % filepath)
        print("Fix license  :")
        if is_llvm_license:
            fix_llvm_license(var)
        else:
            fix_intel_license(var)
    else:
        print("Success      : %s" % filepath)
            
sys.exit(0 if success else 1)
