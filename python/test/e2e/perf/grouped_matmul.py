from functools import partial
from string import Template

import numpy as np

from e2e.reference.grouped_matmul import grouped_matmul_reference

from . import Case


def grouped_matmul_mlir(experts, reduction, columns):
    return Template(
        """
#lhs = affine_map<(e, m, n, k) -> (m, k)>
#weights = affine_map<(e, m, n, k) -> (e, n, k)>
#offsets = affine_map<(e, m, n, k) -> (e)>
#result = affine_map<(e, m, n, k) -> (m, n)>
module {
  func.func @main(%tokens: tensor<?x${reduction}xf16>,
                  %weights: tensor<${experts}x${columns}x${reduction}xf16>,
                  %offsets: tensor<${experts}xi32>) -> tensor<?x${columns}xf16> {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f16
    %rows = tensor.dim %tokens, %c0 : tensor<?x${reduction}xf16>
    %empty = tensor.empty(%rows) : tensor<?x${columns}xf16>
    %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<?x${columns}xf16>)
        -> tensor<?x${columns}xf16>
    %output = linalgx.grouped_matmul <{
        indexing_maps = [#lhs, #weights, #offsets, #result]
      }> ins(%tokens, %weights, %offsets : tensor<?x${reduction}xf16>,
              tensor<${experts}x${columns}x${reduction}xf16>, tensor<${experts}xi32>)
          outs(%init : tensor<?x${columns}xf16>) -> tensor<?x${columns}xf16>
    return %output : tensor<?x${columns}xf16>
  }
}
"""
    ).substitute(experts=experts, reduction=reduction, columns=columns)


def grouped_matmul_inputs(counts, reduction, columns, seed=0):
    rng = np.random.default_rng(seed)
    tokens = (rng.standard_normal((sum(counts), reduction)) * 0.1).astype(np.float16)
    weights = (rng.standard_normal((len(counts), columns, reduction)) * 0.1).astype(
        np.float16
    )
    return tokens, weights, np.cumsum(counts, dtype=np.int32)


def grouped_matmul_outputs(tokens, weights, offsets):
    return (np.empty((tokens.shape[0], weights.shape[1]), tokens.dtype),)


def grouped_matmul_case(name, description, counts, reduction, columns):
    return Case(
        name=name,
        description=description,
        mlir=grouped_matmul_mlir(len(counts), reduction, columns),
        inputs=partial(grouped_matmul_inputs, counts, reduction, columns),
        reference=grouped_matmul_reference,
        rtol=1e-2,
        # Output ~N(0, 0.45^2): f16 rounding of the 2048-term sums.
        atol=1e-2,
        outputs=grouped_matmul_outputs,
    )


CASES = [
    grouped_matmul_case(
        "grouped-matmul-e8",
        "grouped matmul, 8 uneven experts, 2048 tokens, K=2048, N=768, f16",
        (64, 512, 128, 384, 0, 256, 640, 64),
        2048,
        768,
    ),
]
