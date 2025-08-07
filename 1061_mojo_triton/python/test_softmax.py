import numpy as np
import pytest

from test_common import get_test_shapes, load_triton_tensor, load_mojo_tensor


def np_softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


@pytest.mark.parametrize("m, n", get_test_shapes("softmax"))
def test_softmax(m: int, n: int):
    formatted_shape = f"{m:05d}_{n:05d}"
    x_name = f"softmax_x_{formatted_shape}"
    y_name = f"softmax_y_{formatted_shape}"

    triton_x = load_triton_tensor(x_name)
    triton_y = load_triton_tensor(y_name)
    assert (
        triton_x.shape == triton_y.shape == (m, n)
    ), "Unexpected shape for Triton matrix."
    assert (
        triton_x.dtype == triton_y.dtype == np.float16
    ), "Unexpected data type for Triton matrix."

    mojo_x = load_mojo_tensor(x_name)
    mojo_y = load_mojo_tensor(y_name)
    assert mojo_x.shape == mojo_y.shape == (m, n), "Unexpected shape for Mojo matrix."
    assert (
        mojo_x.dtype == mojo_y.dtype == np.float16
    ), "Unexpected data type for Mojo matrix."

    assert np.array_equal(
        triton_x, mojo_x
    ), "Triton and Mojo x matrices aren't identical."

    np_y = np_softmax(triton_x)

    atol, rtol = 1e-2, 1e-2
    assert np.allclose(
        triton_y, np_y, atol=atol, rtol=rtol
    ), "Triton and NumPy y matrices don't match."
    # TODO: Check Mojo output when Mojo kernel is implemented!
    # assert np.allclose(
    #     np_y, mojo_y, atol=atol, rtol=rtol
    # ), "NumPy and Mojo y matrices don't match."
