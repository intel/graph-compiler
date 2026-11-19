import numpy as np


def run_e2e(mlir, inputs, reference, *, rtol, atol, dump_input=False, dump=False):
    import graph_compiler

    if dump_input:
        print(mlir)
    inputs = tuple(np.array(value, copy=True, order="C") for value in inputs)
    expected = reference(*inputs)
    if isinstance(expected, np.ndarray):
        expected = (expected,)
    outputs = tuple(np.full_like(value, np.nan) for value in expected)
    module = graph_compiler.GpuModule(mlir, dump=dump)
    module(*inputs, *outputs)
    for index, (actual, wanted) in enumerate(zip(outputs, expected)):
        np.testing.assert_allclose(
            actual, wanted, rtol=rtol, atol=atol, err_msg=f"Output {index}"
        )
    return outputs