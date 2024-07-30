#ifndef GENIR_OPS
#define GENIR_OPS

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GEN_Dialect : Dialect {
  let name = "gen";
  let cppNamespace = "::mlir::gen";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "gen.kernel"; }
    /// The address space value that represents global memory.
    static constexpr unsigned kGlobalMemoryAddressSpace = 1;
    /// The address space value that represents shared memory.
    static constexpr unsigned kSharedMemoryAddressSpace = 3;
    /// The address space value that represents private memory.
    static constexpr unsigned kPrivateMemoryAddressSpace = 0;
  }];

  let useDefaultAttributePrinterParser = 1;
}

class GEN_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<GEN_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def GEN_TargettAttr : GEN_Attr<"GenTarget", "target"> {
  let description = [{
    GPU target attribute for controlling compilation of targets. All
    parameters decay into default values if not present.

    Examples:

    1. Target with default values.
    ```
      gpu.module @mymodule [#gen.target] attributes {...} {
        ...
      }
    ```
  }];
  let parameters = (ins
    DefaultValuedParameter<"int", "2", "Optimization level to apply.">:$O,
    StringRefParameter<"Target triple.", "\"spirv64-unknown-unknown\"">:$triple,
    StringRefParameter<"Target chip.", "\"xe_1\"">:$chip
  );
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip)^ `>`)?
  }];
  let builders = [
    AttrBuilder<(ins CArg<"int", "2">:$optLevel,
                     CArg<"StringRef", "\"spirv64-unknown-unknown\"">:$triple,
                     CArg<"StringRef", "\"xe_1\"">:$chip), [{
      return Base::get($_ctxt, optLevel, triple, chip);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

#endif // GENIR_OPS
