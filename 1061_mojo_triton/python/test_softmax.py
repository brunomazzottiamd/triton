import numpy as np


def np_softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# TODO: Test corretness with this (AITER error tolerance for fp16):
#       np.allclose(triton_y, np_y, atol=1e-2, rtol=1e-2)
