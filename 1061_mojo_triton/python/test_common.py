from glob import glob
import re

import numpy as np

import np_tensor


TRITON_TENSOR_PREFIX: str = "triton_"
MOJO_TENSOR_PREFIX: str = "mojo___"


def get_test_shapes(tensor_name: str) -> list[int] | list[tuple[int, ...]]:
    split_pattern = re.compile(r"[_\.]")
    root_dir = np_tensor.tensors_dir()
    underscores_in_tensor_name = tensor_name.count("_")
    triton_shapes = {
        tuple(
            int(shape_dim)
            for shape_dim in split_pattern.split(triton_file)[
                3 + underscores_in_tensor_name : -1
            ]
        )
        for triton_file in glob(
            f"{TRITON_TENSOR_PREFIX}{tensor_name}*.npz", root_dir=root_dir
        )
    }
    mojo_shapes = {
        tuple(
            int(shape_dim)
            for shape_dim in split_pattern.split(mojo_file)[
                5 + underscores_in_tensor_name : -1
            ]
        )
        for mojo_file in glob(
            f"{MOJO_TENSOR_PREFIX}{tensor_name}*.npz", root_dir=root_dir
        )
    }
    shapes = triton_shapes & mojo_shapes
    shape_dims = {len(shape) for shape in shapes}
    assert len(shape_dims) == 1, "All shapes must have the same dimension."
    if next(iter(shape_dims)) == 1:
        return sorted(shape[0] for shape in shapes)
    return sorted(shapes)


def load_tensor(tensor_name: str) -> np.ndarray:
    x = np_tensor.load_tensor(tensor_name)
    assert x is not None, f"Unable to load tensor '{tensor_name}'."
    return x


def load_triton_tensor(tensor_name: str) -> np.ndarray:
    return load_tensor(f"{TRITON_TENSOR_PREFIX}{tensor_name}")


def load_mojo_tensor(tensor_name: str) -> np.ndarray:
    return load_tensor(f"{MOJO_TENSOR_PREFIX}{tensor_name}")
