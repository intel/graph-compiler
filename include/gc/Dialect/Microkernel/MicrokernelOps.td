//===- MicrokernelOps.td - microkernel dialect ops ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MICROKERNEL_OPS
#define MICROKERNEL_OPS

include "MicrokernelDialect.td"
include "gc/Dialect/Microkernel/MicrokernelEnum.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class StaticTensorRankOf<list<Type> allowedTypes, list<int> ranks> :
     Type<And<[TensorOf<allowedTypes>.predicate,
              HasAnyRankOfPred<ranks>, HasStaticShapePred]>, 
              !interleave(!foreach(rank, ranks, rank # "D"), "/") # " static " # 
              TensorOf<allowedTypes>.summary, "::mlir::TensorType">;

class StaticMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
     Type<And<[MemRefOf<allowedTypes>.predicate,
              HasAnyRankOfPred<ranks>, HasStaticShapePred]>, 
              !interleave(!foreach(rank, ranks, rank # "D"), "/") # " static " # 
              MemRefOf<allowedTypes>.summary, "::mlir::MemRefType">;

def BrgemmTensor : StaticTensorRankOf<[F32, BF16, SI32, SI8, UI8], [2, 3, 4]>;

def BrgemmTensorOrMemRef : AnyTypeOf<[StaticTensorRankOf<[F32, BF16, SI32, SI8, UI8], [2, 3, 4]>,
                                      StaticMemRefRankOf<[F32, BF16, SI32, SI8, UI8], [2, 3, 4]>]>;

def Microkernel_BrgemmOnTensorOp : Microkernel_Op<"brgemm_on_tensor", 
	[DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         DestinationStyleOpInterface]> {
  let summary = "Abstract Op that execute brgemm kernel on tensors.";
  let description = [{
    The operation has the following arguments: 
    1) Tensors or MemRefs of operand A/B;
    2) The batch dims and leading dims of operand A/B;
    And has the following outputs:
    1) Tensor of operand C;
  }];

  let arguments = (ins Variadic<BrgemmTensorOrMemRef>:$inputs,
                       BrgemmTensorOrMemRef:$init, 
    		       ConfinedAttr<DenseI64ArrayAttr,
                           [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$batchDims,
    		       ConfinedAttr<DenseI64ArrayAttr,
                           [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$leadingDims, 
                       TypedArrayAttrBase<Microkernel_BrgemmFlags, "brgemm flags">:$flags);
  let results = (outs Variadic<BrgemmTensor>:$output);

  let extraClassDeclaration = [{
    Value getOperandA() { return getInputs()[0]; }
    Value getOperandB() { return getInputs()[1]; }
    Value getOperandC() { return getInit(); }
    
    int64_t getBatchDimA() { return getBatchDims()[0]; }
    int64_t getLeadingDimA() { return getLeadingDims()[0]; }
    
    int64_t getBatchDimB() { return getBatchDims()[1]; }
    int64_t getLeadingDimB() { return getLeadingDims()[1]; }

    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
}

def Microkernel_BrgemmDispatchOp : Microkernel_Op<"brgemm.dispatch", [Pure]> {
  let summary = "JIT the brgemm microkernel given the parameters";
  let description = [{
    The operation has the following arguments: 1) m, n, k, lda, ldb, ldc, stride_a and stride_b. 
    Inputs is a dense attribute of I64 elements. 2) flags carry information on
    the different flags that can be used for brgemm like whether beta == 0 or strided batch. For
    more details, see: `Microkernel_BrgemmFlags`. 3) data_types of operand A & B.
    Outpus is the id of JITed kernel.
  }];

  let arguments = (ins
    ConfinedAttr<DenseI64ArrayAttr,
                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$inputs,
    TypedArrayAttrBase<Microkernel_BrgemmFlags, "brgemm flags">:$flags,
    TypedArrayAttrBase<TypeAttr, "brgemm dtypes">:$data_type);

  let results = (outs I64:$results);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Microkernel_BrgemmPrologueOp : Microkernel_Op<"brgemm.prologue"> {
  let summary = "Prologue before executing the JITed brgemm "
                "microkernel, and the context is considered core-level";
  let description = [{
    The operation has the following arguments: Input is the id of JITed kernel.
    There is no output.
  }];

  let arguments = (ins I64:$inputs);

  let assemblyFormat = [{
    `(` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];
}

def Microkernel_BrgemmEpilogueOp : Microkernel_Op<"brgemm.epilogue"> {
  let summary = "Epilogue after executing the JITed brgemm microkernel";
  let description = [{
    The operation has the following arguments: Input is the id of JITed kernel.
    There is no output.
  }];

  let arguments = (ins I64:$inputs);

  let assemblyFormat = [{
    `(` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];
}

/* A generic input type of Microkernel_BrgemmOp, allowing for `BrgemmMemRef` and I64.
 * The `BrgemmMemRef` should be a static MemRef, and for each operand its shape should be:
 *     Operand A: StaticMemRefRankOf<[F32, BF16, SI8, UI8], [3]>;
 *     Operand B (none-VNNI): StaticMemRefRankOf<[F32], [3]>;
 *     Operand B (VNNI): StaticMemRefRankOf<[BF16, SI8, UI8], [4]>;
 *     Operand C: StaticMemRefRankOf<[F32, SI32], [2]>;
 */
def BrgemmMemRefOrI64 : AnyTypeOf<[StaticMemRefRankOf<[F32, BF16, SI32, SI8, UI8], [2, 3, 4]>, I64]>;

def Microkernel_BrgemmOp : Microkernel_Op<"brgemm"> {
  let summary = "execute the JITed brgemm kernel.";
  let description = [{
    The operation has the following arguments: 
    1) For stride mode, id of JITed kernel, MemRef of operand A/B/C, and the batch size;
    2) For addr mode, plus the length of addr list at the end.
    There is no output.
  }];

  let arguments = (ins Variadic<BrgemmMemRefOrI64>:$inputs);

  let assemblyFormat = [{
    `(` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    Value getDispatch() { return getInputs()[0]; }

    Value getOperandA() { return getInputs()[1]; }

    Value getOperandB() { return getInputs()[2]; }

    Value getOutput() { return getInputs()[3]; }

    Value getBatch() { return getInputs()[4]; }

    Value getAddrLen() { return getInputs()[5]; }
  }];

  let hasVerifier = 1;
}

#endif // MICROKERNEL_OPS
