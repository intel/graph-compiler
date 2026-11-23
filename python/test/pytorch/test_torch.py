import graph_compiler

import abc
import pytest
import torch

# import torch.utils.benchmark as benchmark
from typing import Tuple


class TestModule(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, *args: torch.Tensor):
        pass

    @abc.abstractmethod
    def get_inputs(self, dev) -> Tuple[torch.Tensor, ...]:
        pass

    def test(self, rtol=None, atol=None):
        input = self.get_inputs("xpu")
        expect = self(*input)
        if isinstance(expect, torch.Tensor):
            expect = (expect,)
        output = tuple(torch.zeros_like(o) for o in expect)
        gc_mod = self.gc(*input, dump=True)
        gc_mod(*input, *output)
        for o, e in zip(output, expect):
            try:
                passed = False
                torch.testing.assert_close(o, e, rtol=rtol, atol=atol)
                passed = True
            finally:
                if not passed:
                    print(f"Expected:\n{e}")
                    print(f"Actual:\n{o}")


@pytest.mark.parametrize(
    "M,N,DT",
    (
        (2048, 4096, torch.float16),
        (2048, 4096, torch.int32),
    ),
)
def test_arith(M: int, N: int, DT: torch.dtype):

    class Test(TestModule):
        def forward(self, a: torch.Tensor, b: torch.Tensor):
            return ((a + b) * a - b) / a

        def get_inputs(self, dev):
            a = torch.tensor([[2] * M] * N, dtype=DT, device=dev)
            b = torch.tensor([[3] * M] * N, dtype=DT, device=dev)
            return (a, b)

    Test().test()


@pytest.mark.parametrize("M,K,N,DT", ((2048, 8192, 4096, torch.float16),))
def test_matmul(M: int, K: int, N: int, DT: torch.dtype):

    class Test(TestModule):
        def forward(self, a: torch.Tensor, b: torch.Tensor):
            return torch.matmul(a, b)

        def get_inputs(self, dev):
            a = torch.tensor([[3] * K] * M, dtype=DT, device=dev)
            b = torch.tensor([[2] * N] * K, dtype=DT, device=dev)
            return (a, b)

    Test().test()


@pytest.mark.parametrize("M,K,N,DT", ((2048, 8192, 4096, torch.float16),))
def test_matmul_arith(M: int, K: int, N: int, DT: torch.dtype):

    class Test(TestModule):
        def forward(self, a: torch.Tensor, b: torch.Tensor):
            result = torch.matmul(a + a, b + b)
            return (result + result) * result

        def get_inputs(self, dev):
            a = torch.tensor([[3] * K] * M, dtype=DT, device=dev)
            b = torch.tensor([[2] * N] * K, dtype=DT, device=dev)
            return (a, b)

    Test().test()


@pytest.mark.parametrize("M,K,N,DT", ((64, 512, 256, torch.float16),))
def test_swiglu(M: int, K: int, N: int, DT: torch.dtype):

    class Test(TestModule):
        def forward(
            self, tokens: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor
        ):
            gate = torch.matmul(tokens, gate_w)
            up = torch.matmul(tokens, up_w)
            return (gate / (1 + torch.exp(-gate))) * up

        def get_inputs(self, dev):
            torch.manual_seed(0)
            tokens = torch.randn(M, K, dtype=DT, device=dev) * 0.1
            gate_w = torch.randn(K, N, dtype=DT, device=dev) * 0.1
            up_w = torch.randn(K, N, dtype=DT, device=dev) * 0.1
            return (tokens, gate_w, up_w)

    Test().test(rtol=3e-3, atol=2e-4)


def test_sibling_grouped_matmul():
    # torch-mlir cannot lower aten._grouped_mm yet, so compile the grouped
    # operation directly and compare both results with PyTorch on XPU.
    mlir = """
#mapL = affine_map<(e, m, n, k) -> (m, k)>
#mapW = affine_map<(e, m, n, k) -> (e, n, k)>
#mapO = affine_map<(e, m, n, k) -> (e)>
#mapR = affine_map<(e, m, n, k) -> (m, n)>

func.func @main(%lhs: tensor<?x256xf16>, %w0: tensor<8x512x256xf16>,
                %w1: tensor<8x512x256xf16>, %offs: tensor<8xi32>)
    -> (tensor<?x512xf16>, tensor<?x512xf16>) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f16
  %total = tensor.dim %lhs, %c0 : tensor<?x256xf16>
  %dst0 = tensor.empty(%total) : tensor<?x512xf16>
  %dst1 = tensor.empty(%total) : tensor<?x512xf16>
  %init0 = linalg.fill ins(%zero : f16) outs(%dst0 : tensor<?x512xf16>) -> tensor<?x512xf16>
  %init1 = linalg.fill ins(%zero : f16) outs(%dst1 : tensor<?x512xf16>) -> tensor<?x512xf16>
  %a = linalgx.grouped_matmul <{indexing_maps = [#mapL, #mapW, #mapO, #mapR]}>
       ins(%lhs, %w0, %offs : tensor<?x256xf16>, tensor<8x512x256xf16>, tensor<8xi32>)
       outs(%init0 : tensor<?x512xf16>) -> tensor<?x512xf16>
  %b = linalgx.grouped_matmul <{indexing_maps = [#mapL, #mapW, #mapO, #mapR]}>
       ins(%lhs, %w1, %offs : tensor<?x256xf16>, tensor<8x512x256xf16>, tensor<8xi32>)
       outs(%init1 : tensor<?x512xf16>) -> tensor<?x512xf16>
  return %a, %b : tensor<?x512xf16>, tensor<?x512xf16>
}
"""

    experts, rows, k, n = 8, 64, 256, 512
    torch.manual_seed(0)
    tokens = torch.randn(rows, k, dtype=torch.float16, device="xpu") * 0.05
    gate_w = torch.randn(experts, n, k, dtype=torch.float16, device="xpu") * 0.05
    up_w = torch.randn(experts, n, k, dtype=torch.float16, device="xpu") * 0.05
    offsets = torch.arange(1, experts + 1, dtype=torch.int32, device="xpu") * (
        rows // experts
    )

    gate = torch.empty(rows, n, dtype=torch.float16, device="xpu")
    up = torch.empty_like(gate)
    gc_mod = graph_compiler.GpuModule(mlir, dump=True)
    gc_mod(tokens, gate_w, up_w, offsets, gate, up)

    for actual, weights in ((gate, gate_w), (up, up_w)):
        expected = torch._grouped_mm(tokens, weights.transpose(-2, -1), offs=offsets)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize(
    "H,DT",
    (
        (24, torch.float16),
        (16, torch.float16),
    ),
)
def test_concat(H: int, DT: torch.dtype):

    class Test(TestModule):
        def forward(self, a: torch.Tensor, b: torch.Tensor):
            return torch.cat([a, b], dim=2)

        def get_inputs(self, dev):
            a = torch.randn(1, H, 128, 64, dtype=DT, device=dev)
            b = torch.randn(1, H, 1024, 64, dtype=DT, device=dev)
            return (a, b)

    Test().test()


@pytest.mark.parametrize(
    "S0,S1,H,D,DT",
    (
        (1024, 128, 24, 64, torch.float16),
        (512, 256, 16, 64, torch.float16),
    ),
)
def test_transpose_concat(S0: int, S1: int, H: int, D: int, DT: torch.dtype):

    class Test(TestModule):
        def forward(self, a: torch.Tensor, b: torch.Tensor):
            # [1, S0, H, D] -> [1, H, S0, D]
            ta = a.permute(0, 2, 1, 3)
            # [1, S1, H, D] -> [1, H, S1, D]
            tb = b.permute(0, 2, 1, 3)
            return torch.cat([ta, tb], dim=2)

        def get_inputs(self, dev):
            a = torch.randn(1, S0, H, D, dtype=DT, device=dev)
            b = torch.randn(1, S1, H, D, dtype=DT, device=dev)
            return (a, b)

    Test().test()


@pytest.mark.parametrize("M,K,P,Q,N,DT", ((64, 512, 128, 512, 64, torch.float16),))
def test_kernel_depends(M: int, K: int, P: int, Q: int, N: int, DT: torch.dtype):

    class Test(TestModule):
        def forward(
            self,
            a: torch.Tensor,
            b: torch.Tensor,
            c: torch.Tensor,
            d: torch.Tensor,
        ):
            return torch.matmul(torch.matmul(a, b), torch.matmul(c, d))

        def get_inputs(self, dev):
            a = torch.full((M, K), 0.01, dtype=DT, device=dev)
            b = torch.full((K, P), 0.01, dtype=DT, device=dev)
            c = torch.full((P, Q), 0.01, dtype=DT, device=dev)
            d = torch.full((Q, N), 0.01, dtype=DT, device=dev)
            return (a, b, c, d)

    Test().test()
