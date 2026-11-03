import pytest
import torch

import graph_compiler as gc


def initial_value(branch, shared):
    # Shared-LHS cases also exercise a shared initialization producer, including
    # two shared initializers plus a distinct third initializer.
    return 0.25 if shared == "lhs" and branch == 1 else 0.25 * (branch + 1)


def grouped_matmul_mlir(branches, shared, experts, n, k, offsets_dtype, bound):
    """Build explicit fused IR; automatic sibling-op fusion is a separate pass."""
    lhs_type = f"tensor<?x{k}xf16>"
    weights_type = f"tensor<{experts}x{n}x{k}xf16>"
    offsets_type = f"tensor<{experts}x{offsets_dtype}>"
    output_type = f"tensor<?x{n}xf32>"
    args = [f"%a{i}: {lhs_type}" for i in range(branches)]
    args += [f"%w{i}: {weights_type}" for i in range(branches)]
    args += [f"%offs: {offsets_type}"]
    inputs = []
    input_types = []
    for i in range(branches):
        inputs += [
            f"%a{0 if shared == 'lhs' else i}",
            f"%w{0 if shared == 'weights' else i}",
        ]
        input_types += [lhs_type, weights_type]
    inputs += ["%offs"]
    input_types += [offsets_type]
    maps = ["#l"] * branches + ["#w"] * branches + ["#o"] + ["#r"] * branches
    output_types = ", ".join([output_type] * branches)
    results = ", ".join(f"%result#{i}" for i in range(branches))
    bound_init = f"%bound = arith.constant {bound} : index" if bound is not None else ""
    bound_arg = "max_group_size(%bound)" if bound is not None else ""
    init_ops = []
    for i in range(branches):
        init_ops += [
            f"%value{i} = arith.constant {initial_value(i, shared)} : f32",
            f"%empty{i} = tensor.empty(%total) : {output_type}",
            f"%c{i} = linalg.fill ins(%value{i} : f32) outs(%empty{i} : {output_type}) -> {output_type}",
        ]
    return f"""
    #l = affine_map<(e, m, n, k) -> (m, k)>
    #w = affine_map<(e, m, n, k) -> (e, n, k)>
    #o = affine_map<(e, m, n, k) -> (e)>
    #r = affine_map<(e, m, n, k) -> (m, n)>
    module {{
      func.func @main({", ".join(args)}) -> ({output_types}) {{
        %zero = arith.constant 0 : index
        %total = tensor.dim %a0, %zero : {lhs_type}
        {chr(10).join(init_ops)}
        {bound_init}
        %result:{branches} = linalgx.grouped_matmul
            <{{indexing_maps = [{", ".join(maps)}]}}>
            ins({", ".join(inputs)} : {", ".join(input_types)})
            {bound_arg}
            outs({", ".join(f"%c{i}" for i in range(branches))} : {output_types})
            -> ({output_types})
        return {results} : {output_types}
      }}
    }}
    """


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires an Intel GPU")
@pytest.mark.parametrize(
    "branches,shared,k,offsets_dtype,use_bound",
    [
        (1, "lhs", 64, torch.int32, False),
        (2, "lhs", 64, torch.int32, False),
        (3, "lhs", 96, torch.int64, True),
        (2, "weights", 96, torch.int32, True),
    ],
)
def test_fused_grouped_matmul(branches, shared, k, offsets_dtype, use_bound):
    # Use equal, non-empty groups containing full row tiles. Multiple experts
    # need max_group_size so the rectangular grid never schedules zero-row
    # tiles. Without that bound, use one expert so total == group size.
    # A 64-byte-aligned f16 row pitch keeps arbitrary expert starts aligned to
    # the GPU block-load base requirement. K=96 still exercises a K=32 tail.
    groups = [64] * (3 if use_bound else 1)
    total, experts, n = sum(groups), len(groups), 64
    generator = torch.Generator().manual_seed(0)
    lhs = [
        torch.randn(total, k, generator=generator).to(torch.float16) * 0.1
        for _ in range(branches)
    ]
    weights = [
        torch.randn(experts, n, k, generator=generator).to(torch.float16) * 0.1
        for _ in range(branches)
    ]
    if shared == "lhs":
        lhs = [lhs[0]] * branches
    else:
        weights = [weights[0]] * branches
    offsets = torch.tensor(groups, dtype=offsets_dtype).cumsum(0, dtype=offsets_dtype)
    inits = [torch.full((total, n), initial_value(i, shared)) for i in range(branches)]
    expected = [init.clone() for init in inits]
    start = 0
    for expert, rows in enumerate(groups):
        end = start + rows
        for i in range(branches):
            expected[i][start:end] += (
                lhs[i][start:end].float() @ weights[i][expert].float().T
            )
        start = end

    module = gc.GpuModule(
        grouped_matmul_mlir(
            branches,
            shared,
            experts,
            n,
            k,
            "i32" if offsets_dtype == torch.int32 else "i64",
            max(groups) if use_bound else None,
        ),
        dump=True,
    )
    actual = [torch.full_like(init, float("nan")) for init in inits]
    module(*lhs, *weights, offsets, *actual)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=2e-4, rtol=2e-4)
