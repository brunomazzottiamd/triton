from glob import glob
import re

import numpy as np
import pytest

from np_tensor import tensors_dir, load_tensor


def get_vector_add_n() -> list[int]:
    split_pattern = re.compile(r"[_\.]")
    root_dir = tensors_dir()
    triton_ns = {
        int(split_pattern.split(triton_out_file)[4])
        for triton_out_file in glob("triton_vector_add_*.npz", root_dir=root_dir)
    }
    mojo_ns = {
        int(split_pattern.split(triton_out_file)[6])
        for triton_out_file in glob("mojo___vector_add_*.npz", root_dir=root_dir)
    }
    return sorted(triton_ns & mojo_ns)


def load(tensor_name: str) -> np.ndarray:
    x = load_tensor(tensor_name)
    assert x is not None, f"Unable to load tensor '{tensor_name}'."
    return x


@pytest.mark.parametrize("n", get_vector_add_n())
def test_vector_add(n: int):
    triton_x = load(f"triton_vector_add_x_{n:09d}")
    triton_y = load(f"triton_vector_add_y_{n:09d}")
    triton_z = load(f"triton_vector_add_z_{n:09d}")
    assert (
        triton_x.shape == triton_y.shape == triton_z.shape == (n,)
    ), "Unexpected shape for Triton vector."
    assert (
        triton_x.dtype == triton_y.dtype == triton_z.dtype == np.float16
    ), "Unexpected data type for Triton vector."

    mojo_x = load(f"mojo___vector_add_x_{n:09d}")
    mojo_y = load(f"mojo___vector_add_y_{n:09d}")
    mojo_z = load(f"mojo___vector_add_z_{n:09d}")
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
