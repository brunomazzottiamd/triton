import numpy as np

import torch


def gen_tensor(
    shape: int | tuple[int, ...],
    rng_seed: int | None = 20250730,
) -> np.ndarray:
    if rng_seed is not None:
        np.random.seed(rng_seed)
    if isinstance(shape, int):
        shape = (shape,)
    return np.random.randn(*shape).astype(np.float16)


def np_to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).cuda()


def torch_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
