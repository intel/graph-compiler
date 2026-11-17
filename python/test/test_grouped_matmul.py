import sys
from types import SimpleNamespace

import numpy as np
import pytest

from e2e import run_e2e


def grouped_matmul_mlir(experts, reduction, columns):
    return f"""
#lhs = affine_map<(e, m, n, k) -> (m, k)>
#weights = affine_map<(e, m, n, k) -> (e, n, k)>
#offsets = affine_map<(e, m, n, k) -> (e)>
#result = affine_map<(e, m, n, k) -> (m, n)>
module {{
  func.func @main(%tokens: tensor<?x{reduction}xf16>,
                  %weights: tensor<{experts}x{columns}x{reduction}xf16>,
                  %offsets: tensor<{experts}xi32>) -> tensor<?x{columns}xf16> {{
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f16
    %rows = tensor.dim %tokens, %c0 : tensor<?x{reduction}xf16>
    %empty = tensor.empty(%rows) : tensor<?x{columns}xf16>
    %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<?x{columns}xf16>)
        -> tensor<?x{columns}xf16>
    %output = linalgx.grouped_matmul <{{
        indexing_maps = [#lhs, #weights, #offsets, #result]
      }}> ins(%tokens, %weights, %offsets : tensor<?x{reduction}xf16>,
              tensor<{experts}x{columns}x{reduction}xf16>, tensor<{experts}xi32>)
          outs(%init : tensor<?x{columns}xf16>) -> tensor<?x{columns}xf16>
    return %output : tensor<?x{columns}xf16>
  }}
}}
"""


def grouped_matmul_reference(tokens, weights, offsets):
    result = np.empty((tokens.shape[0], weights.shape[1]), dtype=tokens.dtype)
    start = 0
    for expert, end in enumerate(offsets):
        result[start:end] = (
            tokens[start:end].astype(np.float32) @ weights[expert].astype(np.float32).T
        ).astype(tokens.dtype)
        start = int(end)
    return result


def test_grouped_matmul_reference():
    tokens = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float16)
    weights = np.array(
        [[[1, 0], [0, 1]], [[9, 9], [9, 9]], [[2, 0], [0, 3]]],
        dtype=np.float16,
    )
    offsets = np.array([1, 1, 3], dtype=np.int32)
    np.testing.assert_array_equal(
        grouped_matmul_reference(tokens, weights, offsets),
        np.array([[1, 2], [6, 12], [10, 18]], dtype=np.float16),
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_e2e_validation(monkeypatch, corrupt):
    def compile_module(mlir):
        assert mlir == "test module"

        def execute(source, output):
            assert source.flags.c_contiguous
            output[:] = source + int(corrupt)

        return execute

    monkeypatch.setitem(
        sys.modules, "graph_compiler", SimpleNamespace(GpuModule=compile_module)
    )
    source = np.arange(8, dtype=np.float16)[::2]
    if corrupt:
        with pytest.raises(AssertionError, match="Output 0"):
            run_e2e("test module", (source,), np.copy, rtol=0, atol=0)
    else:
        (output,) = run_e2e("test module", (source,), np.copy, rtol=0, atol=0)
        np.testing.assert_array_equal(output, source)


@pytest.mark.gpu
@pytest.mark.parametrize("reduction,columns", [(128, 256), (256, 128)])
@pytest.mark.parametrize(
    "counts",
    [(8, 8, 8, 8), (1, 7, 17, 39), (8, 0, 8, 0), (0, 16, 0, 16), (4, 4, 0, 0)],
    ids=["balanced", "uneven", "empty-middle-last", "empty-first", "empty-tail"],
)
def test_grouped_matmul(counts, reduction, columns):
    pytest.importorskip("graph_compiler")
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
