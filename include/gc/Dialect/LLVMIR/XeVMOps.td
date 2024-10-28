//===-- XeVMOps.td - XeVM dialect definition ---------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef XEVMIR_OPS
#define XEVMIR_OPS

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def XeVM_Dialect : Dialect {
  let name = "xevm";
  let cppNamespace = "::mlir::xevm";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let useDefaultAttributePrinterParser = 1;
}

class XeVM_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<XeVM_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def XeVM_TargettAttr : XeVM_Attr<"XeVMTarget", "target"> {}

class XeVM_Op<string mnemonic, list<Trait> traits = []> :
  Op<XeVM_Dialect, mnemonic, traits>;

def XeVM_ElemType : AnyTypeOf<[AnyI8, AnyI16, AnyI32, F32, F16, BF16]>;

class XeVM_LoadCacheControl<string cacheMnemonic> : I32EnumAttr<!strconcat(cacheMnemonic, "LoadCacheControl"), "XeVM load ops cache control",
  [
    I32EnumAttrCase<"DEFAULT",     0,  "Default">,
    I32EnumAttrCase<"UC",          1,  !strconcat(cacheMnemonic, "UC")>,     // uncached
    I32EnumAttrCase<"C",           2,  !strconcat(cacheMnemonic, "C")>,      // cached
    I32EnumAttrCase<"S",           3,  !strconcat(cacheMnemonic, "S")>,      // streaming
    I32EnumAttrCase<"IAR",         4,  !strconcat(cacheMnemonic, "IAR")>,    // invalidate-after-read
  ]> {
  let cppNamespace = "::mlir::xevm";
}

def XeVM_L1LoadCacheControl : XeVM_LoadCacheControl<"L1">;
def XeVM_L3LoadCacheControl : XeVM_LoadCacheControl<"L3">;

class XeVM_StoreCacheControl<string cacheMnemonic> : I32EnumAttr<!strconcat(cacheMnemonic, "StoreCacheControl"), "XeVM store ops cache control",
  [
    I32EnumAttrCase<"DEFAULT",     0,  "Default">,
    I32EnumAttrCase<"UC",          1,  !strconcat(cacheMnemonic, "UC")>,    // uncached
    I32EnumAttrCase<"WT",          2,  !strconcat(cacheMnemonic, "WT")>,    // write-through
    I32EnumAttrCase<"S",           3,  !strconcat(cacheMnemonic, "S")>,     // streaming
    I32EnumAttrCase<"WB",          4,  !strconcat(cacheMnemonic, "WB")>,    // write back
  ]> {
  let cppNamespace = "::mlir::xevm";
}

def XeVM_L1StoreCacheControl : XeVM_StoreCacheControl<"L1">;
def XeVM_L3StoreCacheControl : XeVM_StoreCacheControl<"L3">;

def XeVM_BlockLoad2dOp : XeVM_Op<"blockload2d">,
  Results<(outs FixedVectorOf<[XeVM_ElemType]>:$res)>,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    I1Attr:$transpose,
    I1Attr:$vnni_transform,
    DefaultValuedAttr<XeVM_L1LoadCacheControl, "::mlir::xevm::L1LoadCacheControl::DEFAULT">:$l1_cache_control,
    DefaultValuedAttr<XeVM_L3LoadCacheControl, "::mlir::xevm::L3LoadCacheControl::DEFAULT">:$l3_cache_control
  )> {

  let summary = "2D block load";

  let description = [{
    The `xevm.blockload2d` operation loads a two dimensional matrix tile
    from a larger matrix residing in memory. The parameters are:
      $ptr - the base address of the matrix containing the tile to load
      $base_width, $base_height, $base_pitch - the shape of matrix
      $x, $y, $tile_width, $tile_height - the starting offsets and shape of the tile to load
      $elem_size_in_bits - the size in bits of the matrix element
        - 32 for f32, bf32
        - 16 for f16, int16, bf16
        - 8 for int8, int4, int2
      $v_blocks - number of tiles to load
      $transpose - transpose the tile in registers (useful for 32 bit element type)
      $vnni_transform - transpose and pack the submatrix in registers (useful for < 32 bit element types)
      $cache_control - an enumerator that sets the L1 and L3 cache behaviour

    Notes:
      - the $transpose and $vnni_transform parameters are mutual exclusive
      - transposing the tile loaded is typically used for the B matrix operand
        (D = C + A * B), where A has row-major layout in registers and B should have column-major layout.
      - if the tile loaded contains out of bound elements of the matrix, they are filled with 0.
      - coordinate is provided in elements, while width and pitch are provided in bytes.
  }];

  let assemblyFormat = [{
    operands ` ` `{` `elem_size_in_bits` `=` $elem_size_in_bits `,` `tile_width` `=` $tile_width `,`
    `tile_height` `=` $tile_height `,` `v_blocks` `=` $v_blocks `,` `transpose` `=` $transpose `,`
    `vnni_transform` `=` $vnni_transform `,` `l1_cache_control` `=` $l1_cache_control `,`
    `l3_cache_control` `=` $l3_cache_control `}` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def XeVM_BlockStore2dOp : XeVM_Op<"blockstore2d">,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemWrite]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    FixedVectorOf<[XeVM_ElemType]>:$stored_val,
    DefaultValuedAttr<XeVM_L1StoreCacheControl, "::mlir::xevm::L1StoreCacheControl::DEFAULT">:$l1_cache_control,
    DefaultValuedAttr<XeVM_L3StoreCacheControl, "::mlir::xevm::L3StoreCacheControl::DEFAULT">:$l3_cache_control
  )> {

  let summary = "2D block store";

  let description = [{
    The `xevm.blockstore2d` operation stores a two dimensional tile into a
    larger matrix residing in memory. The parameters are:
      $ptr - the base address of the matrix where to store the tile
      $base_width, $base_height, $base_pitch - the shape of the matrix
      $x, $y, $tile_width, $tile_height - the starting offsets and shape of the tile to store
      $elem_size_in_bits - the size in bits of the matrix element
        - 32 for f32, bf32
        - 16 for f16, int16, bf16
        - 8 for int8, int4, int2
      $v_blocks - number of tiles to store
      $cache_control - an enumerator that sets the L1 and L3 cache behaviour
      $stored_val - the tile to store

    Notes:
      - coordinate is provided in elements, while width and pitch are provided in bytes.
  }];

  let assemblyFormat = [{
    operands ` ` `{` `elem_size_in_bits` `=` $elem_size_in_bits `,` `tile_width` `=` $tile_width `,`
      `tile_height` `=` $tile_height `,` `v_blocks` `=` $v_blocks `,` `l1_cache_control` `=` $l1_cache_control `,`
      `l3_cache_control` `=` $l3_cache_control `}`
      attr-dict `:` `(` type(operands) `)`
  }];

  let hasVerifier = 1;
}

#endif // XEVMIR_OPS
