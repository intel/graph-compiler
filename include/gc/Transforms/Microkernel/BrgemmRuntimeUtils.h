//===- BrgemmRuntimeUtils.h - Utils for Brgemm Runtime -----------*-C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_MICROKERNEL_BRGEMMRUNTIMEUTILS_H
#define GC_MICROKERNEL_BRGEMMRUNTIMEUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "oneapi/dnnl/dnnl_types.h"

namespace mlir::microkernel {

// these strings contain symbols for BRGEMM interfaces used in mlir pass
static const std::string DNNL_BRGEMM_DISPATCH_NAME = "dnnl_brgemm_dispatch";
static const std::string DNNL_BRGEMM_TILECFG_NAME = "dnnl_brgemm_tileconfig";
static const std::string DNNL_BRGEMM_TILERELEASE_NAME =
    "dnnl_brgemm_tilerelease";
static const std::string DNNL_BRGEMM_EXECUTE_NAME = "dnnl_brgemm_execute";

static inline int64_t getDnnlDataTypeVal(RewriterBase &rewriter,
                                         Attribute attr) {
  auto context = rewriter.getContext();
  auto tattr = dyn_cast_or_null<TypeAttr>(attr);
  assert(tattr);
  if (tattr == TypeAttr::get(FloatType::getF32(context))) {
    return static_cast<int64_t>(dnnl_f32);
  } else if (tattr == TypeAttr::get(FloatType::getBF16(context))) {
    return static_cast<int64_t>(dnnl_bf16);
  } else if (tattr == TypeAttr::get(
                          IntegerType::get(context, 32, IntegerType::Signed))) {
    return static_cast<int64_t>(dnnl_s32);
  } else if (tattr ==
             TypeAttr::get(IntegerType::get(context, 8, IntegerType::Signed))) {
    return static_cast<int64_t>(dnnl_s8);
  } else if (tattr == TypeAttr::get(IntegerType::get(context, 8,
                                                     IntegerType::Unsigned))) {
    return static_cast<int64_t>(dnnl_u8);
  }
  return static_cast<int64_t>(dnnl_data_type_undef);
}

}; // namespace mlir::microkernel

#endif // GC_MICROKERNEL_BRGEMMRUNTIMEUTILS_H
