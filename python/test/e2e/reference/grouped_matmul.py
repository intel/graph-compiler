import numpy as np


def grouped_matmul_reference(tokens, weights, offsets):
    result = np.empty((tokens.shape[0], weights.shape[1]), dtype=tokens.dtype)
    start = 0
    for expert, end in enumerate(offsets):
        result[start:end] = (
            tokens[start:end].astype(np.float32) @ weights[expert].astype(np.float32).T
        ).astype(tokens.dtype)
        start = int(end)
    return result
