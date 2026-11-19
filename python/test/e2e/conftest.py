from functools import partial

import pytest

from e2e.runner import run_e2e as execute_e2e


@pytest.fixture
def run_e2e(pytestconfig):
    return partial(
        execute_e2e,
        dump_input=pytestconfig.getoption("--gc-dump-input"),
        dump=pytestconfig.getoption("--gc-dump"),
    )
