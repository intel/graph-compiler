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
    StringRefParameter<"Target triple.", "\"spir64-unknown-unknown\"">:$triple
  );
  let assemblyFormat = [{
    (`<` struct($O, $triple)^ `>`)?
  }];
  let builders = [
    AttrBuilder<(ins CArg<"int", "2">:$optLevel,
                     CArg<"StringRef", "\"spir64-unknown-unknown\"">:$triple), [{
      return Base::get($_ctxt, optLevel, triple);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

#endif // GENIR_OPS
