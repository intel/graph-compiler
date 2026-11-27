from functools import partial
from string import Template

import numpy as np

from e2e.reference.sdpa import sdpa_reference

from . import Case

# The SDPA fragment, offloaded by the OpenVINO GPU plugin to GC in the SD3.5
# transformer: dynamic batch and sequence, 24 heads of 64, f16.
SDPA_DYN = Template(
    """
module @fragment_name {
  func.func @sdpa(%arg0: memref<?x${heads}x?x${dim}xf16>, %arg1: memref<?x${heads}x?x${dim}xf16>, %arg2: memref<?x${heads}x?x${dim}xf16>, %arg3: memref<?x${heads}x?x${dim}xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<?x${heads}x?x${dim}xf16> to tensor<?x${heads}x?x${dim}xf16>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %0, %c0 : tensor<?x${heads}x?x${dim}xf16>
    %c2 = arith.constant 2 : index
    %dim_0 = tensor.dim %0, %c2 : tensor<?x${heads}x?x${dim}xf16>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<?x${heads}x?x${dim}xf16> to tensor<?x${heads}x?x${dim}xf16>
    %2 = bufferization.to_tensor %arg2 restrict writable : memref<?x${heads}x?x${dim}xf16> to tensor<?x${heads}x?x${dim}xf16>
    %cst = arith.constant ${scale} : f16
    %3 = tensor.empty(%dim, %dim_0) : tensor<?x${heads}x?x${dim}xf16>
    %cst_3 = arith.constant 0.000000e+00 : f16
    %4 = linalg.fill ins(%cst_3 : f16) outs(%3 : tensor<?x${heads}x?x${dim}xf16>) -> tensor<?x${heads}x?x${dim}xf16>
    %5 = linalgx.attention <indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]> ins(%0, %1, %2, %cst : tensor<?x${heads}x?x${dim}xf16>, tensor<?x${heads}x?x${dim}xf16>, tensor<?x${heads}x?x${dim}xf16>, f16) outs(%4 : tensor<?x${heads}x?x${dim}xf16>) -> tensor<?x${heads}x?x${dim}xf16>
    bufferization.materialize_in_destination %5 in restrict writable %arg3 : (tensor<?x${heads}x?x${dim}xf16>, memref<?x${heads}x?x${dim}xf16>) -> ()
    return
  }
}
"""
)


def sdpa_inputs(batch, heads, seq, dim, seed=0):
    rng = np.random.default_rng(seed)
    shape = (batch, heads, seq, dim)
    return tuple(rng.uniform(-1, 1, shape).astype(np.float16) for _ in range(3))


def sdpa_outputs(query, key, value):
    return (np.empty(query.shape, query.dtype),)


def sdpa_case(name, description, batch, heads, seq, dim):
    scale = 1 / np.sqrt(dim)
    return Case(
        name=name,
        description=description,
        mlir=SDPA_DYN.substitute(heads=heads, dim=dim, scale=f"{scale:e}"),
        inputs=partial(sdpa_inputs, batch, heads, seq, dim),
        reference=partial(sdpa_reference, scale=scale),
        rtol=1e-2,
        atol=5e-3,
        outputs=sdpa_outputs,
    )


CASES = [
    # SD3.5 medium at 512x512: 1024 image + 154 text tokens.
    sdpa_case("sdpa-sd35-dyn", "SD3.5 SDPA 2x24x1178x64 f16, dynamic", 2, 24, 1178, 64),
]
