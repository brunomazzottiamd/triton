from glob import glob
import re

import numpy as np
import pytest

from np_tensor import tensors_dir, load_tensor


def get_vector_add_n() -> list[int]:
    split_pattern = re.compile(r"[_\.]")
    root_dir = tensors_dir()
    triton_ns = {
        int(split_pattern.split(triton_out_file)[3])
        for triton_out_file in glob("triton_vector_add_*.npz", root_dir=root_dir)
    }
    mojo_ns = {
        int(split_pattern.split(triton_out_file)[5])
        for triton_out_file in glob("mojo___vector_add_*.npz", root_dir=root_dir)
    }
    return sorted(triton_ns & mojo_ns)


@pytest.mark.parametrize("n", get_vector_add_n())
def test_vector_add(n: int):
    triton_z = load_tensor(f"triton_vector_add_{n:09d}")
    assert triton_z is not None, f"Unable to load Triton output vector for n={n}."
    assert triton_z.shape == (n,), f"Unexpected shape for Triton vector."
    assert triton_z.dtype == np.float16, f"Unexpected data type for Triton vector."

    mojo_z = load_tensor(f"mojo___vector_add_{n:09d}")
    assert mojo_z is not None, f"Unable to load Mojo output vector for n={n}."
    assert mojo_z.shape == (n,), f"Unexpected shape for Mojo vector."
    assert mojo_z.dtype == np.float16, f"Unexpected data type for Mojo vector."
