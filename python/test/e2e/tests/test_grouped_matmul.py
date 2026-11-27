import pytest

from e2e.perf.grouped_matmul import grouped_matmul_inputs, grouped_matmul_mlir
from e2e.reference.grouped_matmul import grouped_matmul_reference


@pytest.mark.gpu
@pytest.mark.parametrize("reduction,columns", [(128, 256), (256, 128)])
@pytest.mark.parametrize(
    "counts",
    [(8, 8, 8, 8), (1, 7, 17, 39), (8, 0, 8, 0), (0, 16, 0, 16), (4, 4, 0, 0)],
    ids=["balanced", "uneven", "empty-middle-last", "empty-first", "empty-tail"],
)
def test_grouped_matmul(run_e2e, counts, reduction, columns):
    run_e2e(
        grouped_matmul_mlir(len(counts), reduction, columns),
        grouped_matmul_inputs(counts, reduction, columns),
        grouped_matmul_reference,
        rtol=1e-2,
        atol=1e-3,
    )
