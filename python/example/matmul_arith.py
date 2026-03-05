import torch
import torch.utils.benchmark as benchmark
import graph_compiler as gc

M = 2048
K = 8192
N = 4096
IT = "f16"
OT = "f32"
IDT = getattr(torch, f"float{IT[1:]}")
ODT = getattr(torch, f"float{OT[1:]}")
MLIR = f"""
module {{
  func.func @main(%arg0: tensor<{M}x{K}x{IT}>, %arg1: tensor<{K}x{N}x{IT}>, %add: tensor<{M}x{K}x{IT}>, %sub: tensor<{M}x{N}x{OT}>) -> tensor<{M}x{N}x{OT}> {{
    %cst = arith.constant 0.000000e+00 : {OT}
    %0 = linalg.add ins(%arg0, %add : tensor<{M}x{K}x{IT}>, tensor<{M}x{K}x{IT}>) outs(%arg0 : tensor<{M}x{K}x{IT}>) -> tensor<{M}x{K}x{IT}>
    %1 = tensor.empty() : tensor<{M}x{N}x{OT}>
    %2 = linalg.fill ins(%cst : {OT}) outs(%1 : tensor<{M}x{N}x{OT}>) -> tensor<{M}x{N}x{OT}>
    %3 = linalg.matmul ins(%0, %arg1 : tensor<{M}x{K}x{IT}>, tensor<{K}x{N}x{IT}>) outs(%2 : tensor<{M}x{N}x{OT}>) -> tensor<{M}x{N}x{OT}>
    %4 = linalg.sub ins(%3, %sub : tensor<{M}x{N}x{OT}>, tensor<{M}x{N}x{OT}>) outs(%1 : tensor<{M}x{N}x{OT}>) -> tensor<{M}x{N}x{OT}>
    %5 = linalg.mul ins(%4, %sub : tensor<{M}x{N}x{OT}>, tensor<{M}x{N}x{OT}>) outs(%1 : tensor<{M}x{N}x{OT}>) -> tensor<{M}x{N}x{OT}>
    return %5 : tensor<{M}x{N}x{OT}>
    // return %3 : tensor<{M}x{N}x{OT}>
  }}
}}
"""


def test():
    dev = "xpu"
    a = torch.tensor([[1] * K] * M, dtype=IDT, device=dev)
    b = torch.tensor([[2] * N] * K, dtype=IDT, device=dev)
    add = torch.tensor([[2] * K] * M, dtype=IDT, device=dev)
    sub = torch.tensor([[5] * N] * M, dtype=ODT, device=dev)
    c = torch.zeros(M, N, dtype=ODT, device=dev)
    mod = gc.GpuModule(MLIR, dump=True, wait=True)
    mod(a, b, add, sub, c)
    expect = (torch.matmul(a + add, b).to(ODT) - sub.to(ODT)) * sub.to(ODT)
    # expect = torch.matmul(a + add, b).to(ODT)
    print(f"Expected:\n{expect}")
    print(f"Actual:\n{c}")
    torch.testing.assert_close(c, expect)


if __name__ == "__main__":
    test()
