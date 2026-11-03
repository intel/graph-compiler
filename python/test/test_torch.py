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
        gc_mod = self.gc(*input, dump=True)
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


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires an Intel GPU")
@pytest.mark.parametrize("shared", ("lhs", "rhs"))
def test_horizontal_fusion(shared, capfd):
    # Static, aligned dimensions keep this focused on fusion rather than edge
    # cases in matrix memory lowering. K=256 also retains a shared K loop.
    m, k, n = 64, 256, 64

    class Test(TestModule):
        def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
            if shared == "lhs":
                return torch.matmul(a, b), torch.matmul(a, c)
            return torch.matmul(a, c), torch.matmul(b, c)

        def get_inputs(self, dev):
            shapes = (
                ((m, k), (k, n), (k, n))
                if shared == "lhs"
                else ((m, k), (m, k), (k, n))
            )
            return tuple(
                torch.randn(shape, dtype=torch.float16, device=dev) * 0.01
                for shape in shapes
            )

    Test().test()
    captured = capfd.readouterr()
    dump = captured.out + captured.err
    # Keep phase dumps visible with pytest -s while also proving that the
    # sibling operations were fused, not merely executed independently.
    print(dump, end="")
    assert "IR Dump HorizontalFusion" in dump
    fusion_ir = dump.split("IR Dump HorizontalFusion", 1)[1].split("IR Dump Tiling", 1)[
        0
    ]
    assert "gc.fused_matmul" in fusion_ir


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires an Intel GPU")
@pytest.mark.parametrize("shared", ("lhs", "weights"))
def test_grouped_horizontal_fusion(shared, capfd):
    # The grouped extension starts from Linalgx IR; PyTorch supplies the
    # reference. Equal groups with a common bound avoid zero-height row tiles.
    a0 = torch.randn(128, 256, dtype=torch.float16) * 0.01
    a1 = a0 if shared == "lhs" else torch.randn_like(a0) * 0.01
    w0 = torch.randn(2, 64, 256, dtype=torch.float16) * 0.01
    w1 = w0 if shared == "weights" else torch.randn_like(w0) * 0.01
    offsets = torch.tensor([64, 128], dtype=torch.int32)
    expected = []
    for a, w in ((a0, w0), (a1, w1)):
        expected.append(
            torch.cat(
                [a[e * 64 : (e + 1) * 64].float() @ w[e].float().T for e in range(2)]
            )
        )
    lhs1 = "%a0" if shared == "lhs" else "%a1"
    weights1 = "%w0" if shared == "weights" else "%w1"
    source = f"""
    #lhs = affine_map<(e, m, n, k) -> (m, k)>
    #weights = affine_map<(e, m, n, k) -> (e, n, k)>
    #offsets = affine_map<(e, m, n, k) -> (e)>
    #out = affine_map<(e, m, n, k) -> (m, n)>
    module {{
      func.func @main(%a0: tensor<128x256xf16>, %a1: tensor<128x256xf16>,
                      %w0: tensor<2x64x256xf16>, %w1: tensor<2x64x256xf16>,
                      %offs: tensor<2xi32>) -> (tensor<128x64xf32>, tensor<128x64xf32>) {{
        %zero = arith.constant 0.0 : f32
        %bound = arith.constant 64 : index
        %empty = tensor.empty() : tensor<128x64xf32>
        %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x64xf32>) -> tensor<128x64xf32>
        %r0 = linalgx.grouped_matmul <{{indexing_maps = [#lhs, #weights, #offsets, #out]}}>
            ins(%a0, %w0, %offs : tensor<128x256xf16>, tensor<2x64x256xf16>, tensor<2xi32>)
            max_group_size(%bound) outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
        %r1 = linalgx.grouped_matmul <{{indexing_maps = [#lhs, #weights, #offsets, #out]}}>
            ins({lhs1}, {weights1}, %offs : tensor<128x256xf16>, tensor<2x64x256xf16>, tensor<2xi32>)
            max_group_size(%bound) outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
        return %r0, %r1 : tensor<128x64xf32>, tensor<128x64xf32>
      }}
    }}
    """
    module = graph_compiler.GpuModule(source, dump=True)
    actual = [torch.zeros_like(value) for value in expected]
    module(a0, a1, w0, w1, offsets, *actual)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)
    captured = capfd.readouterr()
    dump = captured.out + captured.err
    print(dump, end="")
    fusion_ir = dump.split("IR Dump HorizontalFusion", 1)[1].split("IR Dump Tiling", 1)[
        0
    ]
    assert fusion_ir.count(" = linalgx.grouped_matmul") == 1
    assert "gc.fused_grouped_matmul" in dump


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
