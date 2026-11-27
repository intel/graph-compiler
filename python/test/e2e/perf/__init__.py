"""Performance cases: small MLIR fragments, executed end-to-end on the GPU.

Each case is an MLIR module, deterministic inputs, and an independent NumPy
reference. The same cases are used for:

- the correctness tests in `e2e/tests/test_perf_cases.py`;
- the kernel benchmark `python -m e2e.bench`, that measures the device time of
  the kernels with the Level Zero kernel timestamps.

To add a case, create a module in this package, that defines a `CASES` list,
and import it below. The case names must be unique and must not contain `--`.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    mlir: str
    # Returns the input arrays; kernel arguments are the inputs, then outputs.
    inputs: Callable[[], Tuple[np.ndarray, ...]]
    reference: Callable[..., object]
    rtol: float
    atol: float
    # Returns the output arrays, defining their shapes and types, for the
    # benchmarks without the result check. Defaults to the reference results.
    outputs: Optional[Callable[..., Tuple[np.ndarray, ...]]] = None


def _all_cases():
    from . import grouped_matmul, sdpa

    cases = [*sdpa.CASES, *grouped_matmul.CASES]
    names = [case.name for case in cases]
    assert len(names) == len(set(names)), f"Duplicate case names: {names}"
    assert not any("--" in name for name in names), names
    return cases


CASES = _all_cases()


def select(patterns=()):
    """Return the cases whose name matches any of the regexes, all by default."""
    import re

    if not patterns:
        return list(CASES)
    return [c for c in CASES if any(re.search(p, c.name) for p in patterns)]
