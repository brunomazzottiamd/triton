import numpy as np
import torch


def np_to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).cuda()


def torch_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
