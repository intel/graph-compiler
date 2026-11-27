import numpy as np


def sdpa_reference(query, key, value, scale):
    """softmax(scale * Q @ K^T) @ V over the [batch, heads, seq, dim] arrays,
    computed in f32."""
    result = np.empty(query.shape[:-1] + value.shape[-1:], dtype=query.dtype)
    # One batch at a time to bound the memory of the scores.
    for batch in range(query.shape[0]):
        q = query[batch].astype(np.float32)
        k = key[batch].astype(np.float32)
        v = value[batch].astype(np.float32)
        scores = np.matmul(q, k.transpose(0, 2, 1)) * np.float32(scale)
        scores -= scores.max(axis=-1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=-1, keepdims=True)
        result[batch] = np.matmul(scores, v).astype(query.dtype)
    return result
