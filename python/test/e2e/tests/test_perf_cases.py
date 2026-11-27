import pytest

from e2e.perf import CASES


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_perf_case(run_e2e, case):
    run_e2e(case.mlir, case.inputs(), case.reference, rtol=case.rtol, atol=case.atol)
