//===- OneDNNGraphDialect.td - OneDNN input dialect --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ONEDNNGRAPH_DIALECT
#define ONEDNNGRAPH_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// OneDNNGraph dialect definition.
//===----------------------------------------------------------------------===//

def OneDNNGraphDialect : Dialect {
    let name = "onednn_graph";
    let summary = "A dialect for oneDNN Graph.";
    let description = [{
        This dialect follows oneDNN Graph Specification.
    }];
    let cppNamespace = "::mlir::onednn_graph";
}

#endif // ONEDNNGRAPH_DIALECT
