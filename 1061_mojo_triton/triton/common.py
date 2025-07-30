import os

import numpy as np

import torch


DEFAULT_RNG_SEED: int = 20250730


def gen_tensor(
    shape: int | tuple[int, ...],
    rng_seed: int | None = DEFAULT_RNG_SEED,
) -> np.ndarray:
    if rng_seed is not None:
        np.random.seed(rng_seed)
    if isinstance(shape, int):
        shape = (shape,)
    return np.random.randn(*shape).astype(np.float16)


def tensor_equal(x: np.ndarray, y: np.ndarray) -> bool:
    return np.array_equal(x, y)


def tensors_dir() -> str:
    tensors_dir = os.path.join(os.getcwd(), "tensors")
    if not os.path.exists(tensors_dir):
        os.mkdir(tensors_dir)
    return tensors_dir


def tensor_file(tensor_name: str) -> str:
    return os.path.join(tensors_dir(), f"{tensor_name}.npz")


def save_tensor(tensor_name: str, x: np.ndarray) -> None:
    np.savez_compressed(tensor_file(tensor_name), x)


def load_tensor(tensor_name: str) -> np.ndarray | None:
    f = tensor_file(tensor_name)
    if not os.path.exists(f):
        return None
    return np.load(f)["arr_0"]


def np_to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).cuda()


def torch_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
