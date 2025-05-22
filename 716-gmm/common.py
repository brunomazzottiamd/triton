# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------


# PyTorch
import torch
from torch import Tensor

# Triton
import triton


# Global defaults.
# ------------------------------------------------------------------------------


# Default device.
DEVICE: torch.device | str = "cuda"


# Default RNG seed for input generation.
RNG_SEED: int = 0


# Default number of group sizes to use when benchmarking and launching the kernel for profiling.
NUM_GROUP_SIZES: int = 1


# Defaut tiling.


def is_power_of_2(x: int) -> bool:
    return (x > 0) and (x & (x - 1) == 0)


TILING: tuple[int, int, int] = (64, 64, 64)
assert all(
    is_power_of_2(tiling_dim) for tiling_dim in TILING
), "Invalid default tiling."


# Default transposition (TN).
TRANS_LHS: bool = False
TRANS_RHS: bool = True
TRANS_OUT: bool = False


# Real GMM shapes, used by real models.
# fmt: off
REAL_SHAPES: list[tuple[int, int, int, int]] = [
    #      M,     K,     N,   G
    (  49152,  1408,  2048,  64),  # deepseekv2-16B
    (3145728,  2048,  1408,   8),  # deepseekv2-16B (IT'S BIG! I was getting core dump with this shape! lhs => 12 GB, out => 8.25 GB)
    ( 393216,  2048,  1408,  64),  # deepseekv2-16B
    (  32768,  6144, 16384,   8),  # Mixtral 8x22B proxy model
    (  32768, 16384,  6144,   8),  # Mixtral 8x22B proxy model
]
# fmt: on


# Hardware capabilities.
# ------------------------------------------------------------------------------


def num_sms(device: torch.device | str = DEVICE) -> int:
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    assert num_sms, f"Number of SMs must be positive (it's {num_sms})."
    return num_sms


# Parameter checking functions.
# ------------------------------------------------------------------------------


def check_input_device_dtype(lhs: Tensor, rhs: Tensor, group_sizes: Tensor) -> None:
    assert (
        lhs.device == rhs.device == group_sizes.device
    ), f"All input tensors must be in the same device (lhs = {lhs.device}, rhs = {rhs.device}, group_sizes = {group_sizes.device})."
    assert (
        lhs.dtype == rhs.dtype
    ), f"lhs and rhs types must match (lhs = {lhs.dtype}, rhs = {rhs.dtype})."
    assert group_sizes.dtype == torch.int32, "group_sizes type must be int32."


# Functions to extract information from generated tensors.
# ------------------------------------------------------------------------------


def get_tiling(
    M: int,
    K: int,
    N: int,
    tiling: tuple[int, int, int],
    group_sizes: Tensor | None = None,
) -> tuple[int, int, int]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert len(tiling) == 3, f"tiling must have 3 dimensions (it's = {len(tiling)})."
    if group_sizes is not None:
        max_group_size = int(torch.max(group_sizes).item())
        assert (
            max_group_size > 0
        ), f"The size of the largest group must be positive (it's {max_group_size})."
        M = min(M, max_group_size)

    block_size_m, block_size_k, block_size_n = tiling

    # Pick smaller block sizes for toy shapes.
    block_size_m = min(triton.next_power_of_2(M), block_size_m)
    block_size_k = min(triton.next_power_of_2(K), block_size_k)
    block_size_n = min(triton.next_power_of_2(N), block_size_n)

    assert is_power_of_2(
        block_size_m
    ), f"M-dimension tile size must be a power of 2 (it's {block_size_m})."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."

    return block_size_m, block_size_k, block_size_n
