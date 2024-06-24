################################################################################
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
# SPDX-License-Identifier: Apache-2.0
################################################################################

if(NOT GC_BENCH_ENABLE)
    message(STATUS "Benchgc is not enabled")
    return()
endif()

configure_file(setup.py ${CMAKE_BINARY_DIR}/test/benchgc/setup.py COPYONLY)

file(GLOB PYTHON_SCRIPTS "*.py")
foreach(PY_SCRIPT ${PYTHON_SCRIPTS})
  configure_file(${PY_SCRIPT} ${CMAKE_BINARY_DIR} COPYONLY)
endforeach()

add_custom_target(benchgc
    COMMAND ${Python_EXECUTABLE} setup.py bdist_wheel
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test/benchgc/"
    DEPENDS GcPythonModules) 

add_subdirectory("src/benchgc")
add_subdirectory("src/benchgc/ops")
add_subdirectory("src/benchgc/fill")
add_subdirectory("src/benchgc/linalg")