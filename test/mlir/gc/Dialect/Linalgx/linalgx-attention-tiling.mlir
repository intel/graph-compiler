// RUN: gc-opt --transform-interpreter --split-input-file %s | FileCheck %s

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalgx.attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0 num_threads [8, 16]
         : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @attention_f16(%query: tensor<192x1024x64xf16>,
                         %key: tensor<192x1024x64xf16>,
                         %value: tensor<192x1024x64xf16>,
                         %output: tensor<192x1024x64xf32>)
                         -> (tensor<192x1024x64xf32>) {
  %scale = arith.constant 1.0 : f16

  %out = linalgx.attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO] }
        ins(%query, %key, %value, %scale : tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, f16)
        outs(%output : tensor<192x1024x64xf32>)
        -> tensor<192x1024x64xf32>

  return %out : tensor<192x1024x64xf32>
}

// CHECK: scf.forall
// CHECK: linalgx.attention
// CHECK: scf.forall.in_parallel
