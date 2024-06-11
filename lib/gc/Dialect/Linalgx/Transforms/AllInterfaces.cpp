//===-- AllInterfaces.cpp - linalgx dialect interfaces ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/Transforms/AllInterfaces.h"

#include "gc/Dialect/Linalgx/Transforms/BufferizableOpInterfaceImpl.h"
#include "gc/Dialect/Linalgx/Transforms/TilingInterfaceImpl.h"

void mlir::linalgx::registerAllDialectInterfaceImplementations(
    DialectRegistry &registry) {
  registerBufferizableOpInterfaceExternalModels(registry);
  registerTilingInterfaceExternalModels(registry);
}
