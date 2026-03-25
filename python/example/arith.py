import torch
import torch.utils.benchmark as benchmark
import graph_compiler as gc

M = 2048
N = 4096
T = "f16"
DT = getattr(torch, f"{'float' if T[0] == 'f' else 'int'}{T[1:]}")
MLIR = f"""
module {{
  func.func @main(%arg0: tensor<{M}x{N}x{T}>, %arg1: tensor<{M}x{N}x{T}>) -> tensor<{M}x{N}x{T}> {{
    %cst = arith.constant 0.000000e+00 : {T}
    %1 = tensor.empty() : tensor<{M}x{N}x{T}>
    %2 = linalg.fill ins(%cst : {T}) outs(%1 : tensor<{M}x{N}x{T}>) -> tensor<{M}x{N}x{T}>
    %3 = linalg.add ins(%arg0, %arg1 : tensor<{M}x{N}x{T}>, tensor<{M}x{N}x{T}>) outs(%2 : tensor<{M}x{N}x{T}>) -> tensor<{M}x{N}x{T}>
    %4 = linalg.mul ins(%3, %arg0 : tensor<{M}x{N}x{T}>, tensor<{M}x{N}x{T}>) outs(%1 : tensor<{M}x{N}x{T}>) -> tensor<{M}x{N}x{T}>
    %5 = linalg.sub ins(%4, %arg1 : tensor<{M}x{N}x{T}>, tensor<{M}x{N}x{T}>) outs(%1 : tensor<{M}x{N}x{T}>) -> tensor<{M}x{N}x{T}>
    %6 = linalg.div ins(%5, %arg0 : tensor<{M}x{N}x{T}>, tensor<{M}x{N}x{T}>) outs(%1 : tensor<{M}x{N}x{T}>) -> tensor<{M}x{N}x{T}>
    return %6 : tensor<{M}x{N}x{T}>
  }}
}}
"""


class ArithModel(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return ((a + b) * a - b) / a

    @staticmethod
    def get_inputs(dev):
        a = torch.tensor([[2] * M] * N, dtype=DT, device=dev)
        b = torch.tensor([[3] * M] * N, dtype=DT, device=dev)
        return (a, b)


def test_mlir():
    dev = "xpu"
    a = torch.tensor([[2] * M] * N, dtype=DT, device=dev)
    b = torch.tensor([[3] * M] * N, dtype=DT, device=dev)
    c = torch.zeros_like(a)
    mod = gc.GpuModule(MLIR, dump=True, wait=True)
    mod(a, b, c)
    expect = ((a + b) * a - b) / a
    print(f"Expected:\n{expect}")
    print(f"Actual:\n{c}")
    torch.testing.assert_close(c, expect)


def test_torch():
    torch_mod = ArithModel()
    input = torch_mod.get_inputs("xpu")
    expect = torch_mod(*input)
    output = torch.zeros_like(expect)
    gc_mod = torch_mod.gc(*input, dump=True, wait=True)
    gc_mod(*input, output)
    print(f"Expected:\n{expect}")
    print(f"Actual:\n{output}")
    torch.testing.assert_close(output, expect)


if __name__ == "__main__":
    test_mlir()
    test_torch()
