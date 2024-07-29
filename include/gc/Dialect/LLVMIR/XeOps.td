#ifndef XEIR_OPS
#define XEIR_OPS

// include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"


// def XE_TargettAttr : Xe_Attr<"XeTarget", "target"> {
//   let description = [{
//     GPU target attribute for controlling compilation of targets. All
//     parameters decay into default values if not present.

//     Examples:

//     1. Target with default values.
//     ```
//       gpu.module @mymodule [#xe.target] attributes {...} {
//         ...
//       }
//     ```
//   }];
//   let parameters = (ins
//     DefaultValuedParameter<"int", "2", "Optimization level to apply.">:$O,
//     StringRefParameter<"Target triple.", "\"spir64-unknown-unknown\"">:$triple
//   );
//   let assemblyFormat = [{
//     (`<` struct($O, $triple)^ `>`)?
//   }];
//   let builders = [
//     AttrBuilder<(ins CArg<"int", "2">:$optLevel,
//                      CArg<"StringRef", "\"spir64-unknown-unknown\"">:$triple), [{
//       return Base::get($_ctxt, optLevel, triple);
//     }]>
//   ];
//   let skipDefaultBuilders = 1;
//   let genVerifyDecl = 1;
// }

#endif // XEIR_OPS
