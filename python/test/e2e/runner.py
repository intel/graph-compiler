import statistics
import time

import numpy as np


def _contiguous(values):
    return tuple(np.array(value, copy=True, order="C") for value in values)


def _expected(reference, inputs):
    expected = reference(*inputs)
    return (expected,) if isinstance(expected, np.ndarray) else tuple(expected)


def _compile(mlir, dump_input, dump):
    import graph_compiler

    if dump_input:
        print(mlir)
    return graph_compiler.GpuModule(mlir, dump=dump)


def _check(module, inputs, expected, rtol, atol):
    outputs = tuple(np.full_like(value, np.nan) for value in expected)
    module(*inputs, *outputs)
    for index, (actual, wanted) in enumerate(zip(outputs, expected)):
        np.testing.assert_allclose(
            actual, wanted, rtol=rtol, atol=atol, err_msg=f"Output {index}"
        )
    return outputs


def run_e2e(mlir, inputs, reference, *, rtol, atol, dump_input=False, dump=False):
    inputs = _contiguous(inputs)
    expected = _expected(reference, inputs)
    module = _compile(mlir, dump_input, dump)
    return _check(module, inputs, expected, rtol, atol)


def stats_ms(values_ns):
    """min/median/mean/stdev/max of the nanosecond values, in milliseconds."""
    ms = [v / 1e6 for v in values_ns]
    return {
        "min": min(ms),
        "median": statistics.median(ms),
        "mean": statistics.fmean(ms),
        "stdev": statistics.stdev(ms) if len(ms) > 1 else 0.0,
        "max": max(ms),
    }


def bench_e2e(
    mlir,
    inputs,
    reference,
    *,
    rtol,
    atol,
    outputs=None,
    warmup=50,
    iters=200,
    check=True,
    dump_input=False,
    dump=False,
):
    """Compile the MLIR, optionally check the results against the reference, and
    measure the program execution time.

    The arguments are copied to the device once, then the program is executed
    `warmup + iters` times on the same device buffers, waiting for completion
    after each run. The device time of each run is the sum of its kernels'
    execution times, measured with the Level Zero kernel timestamps, so it
    excludes the launch overhead and the host-device copies. The host time is
    the wall-clock time of the whole run, including the launch and the wait.

    Without `check`, the reference is not called if the `outputs` arrays, that
    define the output shapes and types, are provided.

    Returns a dict with the `device_ms` and `host_ms` statistics per run, and
    the per-kernel statistics in `kernels`.
    """
    import graph_compiler

    if warmup < 0 or iters <= 0:
        raise ValueError(f"Invalid warmup={warmup} or iters={iters}")
    inputs = _contiguous(inputs)
    if check or outputs is None:
        expected = _expected(reference, inputs)
    else:
        expected = _contiguous(outputs)
    module = _compile(mlir, dump_input, dump)
    if check:
        _check(module, inputs, expected, rtol, atol)

    args = [graph_compiler.Usm(value) for value in (*inputs, *expected)]
    profiling = graph_compiler.is_profiling()
    graph_compiler.set_profiling(True)
    try:
        graph_compiler.take_kernel_times()
        for _ in range(warmup):
            module(*args)
        graph_compiler.take_kernel_times()

        host, device, kernels = [], [], {}
        for _ in range(iters):
            start = time.perf_counter_ns()
            module(*args)
            host.append(time.perf_counter_ns() - start)
            times = graph_compiler.take_kernel_times()
            device.append(sum(ns for _, ns in times))
            per_run = {}
            for name, ns in times:
                per_run.setdefault(name, []).append(ns)
            for name, values in per_run.items():
                kernels.setdefault(name, []).append(values)
    finally:
        graph_compiler.set_profiling(profiling)

    if not any(device):
        raise RuntimeError("No kernel execution times were collected")
    return {
        "iters": iters,
        "warmup": warmup,
        "checked": check,
        "device_ms": stats_ms(device),
        "host_ms": stats_ms(host),
        # Launches per run and the statistics of their total time per run.
        "kernels": {
            name: {
                "launches": max(len(v) for v in runs),
                **stats_ms([sum(v) for v in runs]),
            }
            for name, runs in sorted(kernels.items())
        },
    }
