from string import Template

import numpy as np
import pytest

from e2e.reference.grouped_matmul import grouped_matmul_reference


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


@pytest.mark.gpu
@pytest.mark.parametrize("reduction,columns", [(128, 256), (256, 128)])
@pytest.mark.parametrize(
    "counts",
    [(8, 8, 8, 8), (1, 7, 17, 39), (8, 0, 8, 0), (0, 16, 0, 16), (4, 4, 0, 0)],
    ids=["balanced", "uneven", "empty-middle-last", "empty-first", "empty-tail"],
)
def test_grouped_matmul(run_e2e, counts, reduction, columns):
    rng = np.random.default_rng(0)
    tokens = (rng.standard_normal((sum(counts), reduction)) * 0.1).astype(np.float16)
    weights = (rng.standard_normal((len(counts), columns, reduction)) * 0.1).astype(
        np.float16
    )
    offsets = np.cumsum(counts, dtype=np.int32)
    run_e2e(
        grouped_matmul_mlir(len(counts), reduction, columns),
        (tokens, weights, offsets),
        grouped_matmul_reference,
        rtol=1e-2,
        atol=1e-3,
    )