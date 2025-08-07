import numpy as np
import pytest

from test_common import get_test_shapes, load_triton_tensor, load_mojo_tensor


@pytest.mark.parametrize("n", get_test_shapes("vector_add"))
def test_vector_add(n: int):
    formatted_n = f"{n:09d}"
    x_name = f"vector_add_x_{formatted_n}"
    y_name = f"vector_add_y_{formatted_n}"
    z_name = f"vector_add_z_{formatted_n}"

    triton_x = load_triton_tensor(x_name)
    triton_y = load_triton_tensor(y_name)
    triton_z = load_triton_tensor(z_name)
    assert (
        triton_x.shape == triton_y.shape == triton_z.shape == (n,)
    ), "Unexpected shape for Triton vector."
    assert (
        triton_x.dtype == triton_y.dtype == triton_z.dtype == np.float16
    ), "Unexpected data type for Triton vector."

    mojo_x = load_mojo_tensor(x_name)
    mojo_y = load_mojo_tensor(y_name)
    mojo_z = load_mojo_tensor(z_name)
    assert (
        mojo_x.shape == mojo_y.shape == mojo_z.shape == (n,)
    ), "Unexpected shape for Mojo vector."
    assert (
        mojo_x.dtype == mojo_y.dtype == mojo_z.dtype == np.float16
    ), "Unexpected data type for Mojo vector."

    assert np.array_equal(
        triton_x, mojo_x
    ), "Triton and Mojo x vectors aren't identical."
    assert np.array_equal(
        triton_y, mojo_y
    ), "Triton and Mojo y vectors aren't identical."

    np_z = triton_x + triton_y

    assert np.array_equal(
        triton_z, mojo_z
    ), "Triton and Mojo z vectors aren't identical."
    assert np.array_equal(
        mojo_z,
        np_z,
    ), "Mojo and NumPy z vectors aren't identical."
