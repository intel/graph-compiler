import torch
import torch.utils.benchmark as benchmark
import graph_compiler as gc

MLIR = """
module {
  func.func @main(%arg0: tensor<2048x8192xf16>, %arg1: tensor<8192x4096xf16>) -> tensor<2048x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<2048x4096xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2048x4096xf32>) -> tensor<2048x4096xf32>
    %3 = linalg.matmul ins(%arg0, %arg1 : tensor<2048x8192xf16>, tensor<8192x4096xf16>) outs(%2 : tensor<2048x4096xf32>) -> tensor<2048x4096xf32>
    return %3 : tensor<2048x4096xf32>
  }
}
"""


def test():
    dev = "cpu"
    a = torch.tensor([[3.0] * 8192] * 2048, dtype=torch.float16, device=dev)
    b = torch.tensor([[2.0] * 4096] * 8192, dtype=torch.float16, device=dev)
    c = torch.zeros(2048, 4096, dtype=torch.float32, device=dev)
    mod = gc.GpuModule(MLIR, dump=True)
    mod(a, b, c)
    expect = torch.matmul(a, b).to(dtype=torch.float32)
    print(f"Expected:\n{expect}")
    print(f"Actual:\n{c}")
    torch.testing.assert_close(c, expect)


def bench():
    dev = "xpu"
    ta = torch.tensor([[3.0] * 8192] * 2048, dtype=torch.float16, device=dev)
    tb = torch.tensor([[2.0] * 4096] * 8192, dtype=torch.float16, device=dev)
    tc = torch.zeros(2048, 4096, dtype=torch.float16, device=dev)
    tc32 = torch.zeros(2048, 4096, dtype=torch.float32, device=dev)
    ua = ta.usm()
    ub = tb.usm()
    uc = tc32.usm()
    mod = gc.GpuModule(MLIR, dump=True, wait=True)

    # # Warmup
    for _ in range(100):
        mod(ua, ub, uc)
        torch.matmul(ta, tb, out=tc)
        torch.xpu.synchronize()

    # Benchmark torch.matmul
    t_torch = benchmark.Timer(
        stmt='torch.matmul(ta, tb, out=tc); torch.xpu.synchronize()',
        globals={
            'torch': torch,
            'ta': ta,
            'tb': tb,
            'tc': tc
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
