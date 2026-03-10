import torch
import torch.utils.benchmark as benchmark
import graph_compiler as gc

M = 2048
K = 8192
N = 4096
# IT = "i8"
# OT = "i8"
IT = "f16"
OT = "f16"
IDT = getattr(torch, f"{"float" if IT[0] == "f" else "int"}{IT[1:]}")
ODT = getattr(torch, f"{"float" if OT[0] == "f" else "int"}{OT[1:]}")
MLIR = f"""
module {{
  func.func @main(%arg0: tensor<{M}x{K}x{IT}>, %arg1: tensor<{K}x{N}x{IT}>) -> tensor<{M}x{N}x{OT}> {{
    %cst = arith.constant {"0" if OT[0] == "i" else "0.000000e+00"} : {OT}
    %1 = tensor.empty() : tensor<{M}x{N}x{OT}>
    %2 = linalg.fill ins(%cst : {OT}) outs(%1 : tensor<{M}x{N}x{OT}>) -> tensor<{M}x{N}x{OT}>
    %3 = linalg.matmul ins(%arg0, %arg1 : tensor<{M}x{K}x{IT}>, tensor<{K}x{N}x{IT}>) outs(%2 : tensor<{M}x{N}x{OT}>) -> tensor<{M}x{N}x{OT}>
    return %3 : tensor<{M}x{N}x{OT}>
  }}
}}
"""

def test():
    dev = "xpu"
    a = torch.tensor([[3] * K] * M, dtype=IDT, device=dev)
    b = torch.tensor([[2] * N] * K, dtype=IDT, device=dev)
    c = torch.zeros(M, N, dtype=ODT, device=dev)
    mod = gc.GpuModule(MLIR, dump=True, wait=True)
    mod(a, b, c)
    expect = torch.matmul(a, b).to(ODT)
    print(f"Expected:\n{expect}")
    print(f"Actual:\n{c}")
    torch.testing.assert_close(c, expect)


def bench():
    dev = "xpu"
    ta = torch.tensor([[3] * K] * M, dtype=IDT, device=dev)
    tb = torch.tensor([[2] * N] * K, dtype=IDT, device=dev)
    tc = torch.zeros(M, N, dtype=ODT, device=dev)
    ua = ta.usm()
    ub = tb.usm()
    uc = tc.usm()
    mod = gc.GpuModule(MLIR, dump=True, wait=True)

    # # Warmup
    ttc = tc.to(IDT)
    for _ in range(100):
        mod(ua, ub, uc)
        torch.matmul(ta, tb, out=ttc)
        torch.xpu.synchronize()

    # Benchmark torch.matmul
    t_torch = benchmark.Timer(
        stmt='torch.matmul(ta, tb, out=tc); torch.xpu.synchronize()',
        globals={
            'torch': torch,
            'ta': ta,
            'tb': tb,
            'tc': ttc,
        })

    # Benchmark graph_compiler
    t_gc = benchmark.Timer(stmt='mod(ua, ub, uc)',
                           globals={
                               'mod': mod,
                               'ua': ua,
                               'ub': ub,
                               'uc': uc
                           })

    print("Graph Compiler:", t_gc.timeit(100))
    print("PyTorch matmul:", t_torch.timeit(100))


if __name__ == "__main__":
    test()
    bench()
