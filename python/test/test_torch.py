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

    def test(self):
        input = self.get_inputs("xpu")
        expect = self(*input)
        if isinstance(expect, torch.Tensor):
            expect = (expect,)
        output = tuple(torch.zeros_like(o) for o in expect)
        gc_mod = self.gc(*input, dump=True, wait=True)
        gc_mod(*input, *output)
        for o, e in zip(output, expect):
            try:
                passed = False
                torch.testing.assert_close(o, e)
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
