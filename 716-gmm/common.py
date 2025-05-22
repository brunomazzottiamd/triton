# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import logging

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


# Probabilities for generating random group sizes.
UNUSED_TOKENS_PROB: float = 0.0
UNUSED_EXPERTS_PROB: float = 0.1


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


# Generation of group sizes.
# ------------------------------------------------------------------------------


def gen_uniform_group_sizes(
    M: int,
    G: int,
    device: torch.device | str = DEVICE,
) -> Tensor:
    assert M >= 0, f"Number of tokens M must be non-negative (it's {M})."
    assert G > 0, f"Number of experts G must be positive (it's {G})."

    base = M // G
    remainder = M % G
    group_sizes = torch.full((G,), base, dtype=torch.int32, device=device)
    if remainder > 0:
        group_sizes[:remainder] += 1

    assert (
        len(group_sizes) == G
    ), f"Group sizes don't have {G} elements (it's {len(group_sizes)})."
    assert torch.all(group_sizes >= 0).item(), "All group sizes must be non-negative."
    assert (
        torch.sum(group_sizes).item() == M
    ), f"Group sizes don't add up to total tokens {M}."
    assert group_sizes.dtype == torch.int32, "Group sizes must be int32."

    return group_sizes


def gen_group_sizes(
    M: int,
    G: int,
    device: torch.device | str = DEVICE,
    rng_seed: int | None = RNG_SEED,
    unused_tokens_prob: float = UNUSED_TOKENS_PROB,
    unused_experts_prob: float = UNUSED_EXPERTS_PROB,
) -> Tensor:
    assert M >= 0, f"Number of tokens M must be non-negative (it's {M})."
    assert G > 0, f"Number of experts G must be positive (it's {G})."
    assert (
        0 <= unused_tokens_prob <= 1
    ), f"Probability of unused tokens must be in [0, 1] interval (it's {unused_tokens_prob})."
    assert (
        0 <= unused_experts_prob <= 1
    ), f"Probability of unused experts must be in [0, 1] interval (it's {unused_experts_prob})."

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    if unused_tokens_prob > 0:
        # Optionally drop tokens to simulate routing sparsity, some tokens may not be routed.
        num_unused_tokens = M
        while num_unused_tokens == M:
            num_unused_tokens = int(
                torch.binomial(
                    torch.tensor(float(M), device=device),
                    torch.tensor(unused_tokens_prob, device=device),
                ).item()
            )
    else:
        num_unused_tokens = 0
    num_used_tokens = M - num_unused_tokens
    assert (
        num_unused_tokens >= 0
    ), f"Number of unused tokens must be non-negative (it's {num_unused_tokens})."
    assert (
        num_used_tokens > 0
    ), f"Number of used tokens must be positive (it's {num_used_tokens})."
    assert (
        num_used_tokens + num_unused_tokens == M
    ), f"Unused + used tokens don't add up total tokens ({num_used_tokens} + {num_unused_tokens} != {M})."

    if num_unused_tokens > 0:
        logging.debug(
            "Group sizes generation: dropped %d token%s.",
            num_unused_tokens,
            "s" if num_unused_tokens > 1 else "",
        )

    if unused_experts_prob > 0:
        # Some experts may have zero tokens assigned to them.
        num_used_experts = 0
        while num_used_experts == 0:
            used_experts = torch.nonzero(
                torch.rand((G,), device=device) >= unused_experts_prob
            ).squeeze()
            num_used_experts = used_experts.numel()
    else:
        used_experts = torch.arange(0, G, device=device)
        num_used_experts = G
    num_unused_experts = G - num_used_experts
    assert (
        num_unused_experts >= 0
    ), f"Number of unused experts must be non-negative (it's {num_unused_experts})."
    assert (
        num_used_experts >= 1
    ), f"At least one expert must be used (it's {num_used_experts})."
    assert (
        num_unused_experts + num_used_experts == G
    ), f"Unused + used experts don't add up total experts ({num_unused_experts} + {num_used_experts} != {G})."

    if num_unused_experts > 0:
        logging.debug(
            "Group sizes generation: dropped %d expert%s.",
            num_unused_experts,
            "s" if num_unused_experts > 1 else "",
        )

    group_sizes = torch.bincount(
        used_experts[
            torch.randint(low=0, high=num_used_experts, size=(num_used_tokens,))
        ],
        minlength=G,
    ).to(torch.int32)

    assert (
        len(group_sizes) == G
    ), f"Group sizes don't have {G} elements (it's {len(group_sizes)})."
    assert torch.all(group_sizes >= 0).item(), "All group sizes must be non-negative."
    assert (
        torch.sum(group_sizes).item() == num_used_tokens
    ), f"Group sizes don't add up to used tokens {num_used_tokens}."
    assert group_sizes.dtype == torch.int32, "Group sizes must be int32."

    return group_sizes


def gen_multiple_group_sizes(
    num_group_sizes: int,
    M: int,
    G: int,
    device: torch.device | str = DEVICE,
    rng_seed: int | None = RNG_SEED,
    unused_tokens_prob: float = UNUSED_TOKENS_PROB,
    unused_experts_prob: float = UNUSED_EXPERTS_PROB,
    group_sizes_0: Tensor | None = None,
) -> list[Tensor]:
    assert (
        num_group_sizes > 0
    ), f"Number of group sizes to be generated must be positive, it's {num_group_sizes}."
    multiple_group_sizes = [
        gen_group_sizes(
            M,
            G,
            device=device,
            rng_seed=rng_seed if g == 0 else None,
            unused_tokens_prob=unused_tokens_prob,
            unused_experts_prob=unused_experts_prob,
        )
        for g in range(
            num_group_sizes if group_sizes_0 is None else num_group_sizes - 1
        )
    ]
    if group_sizes_0 is not None:
        multiple_group_sizes.insert(0, group_sizes_0)
    assert (
        len(multiple_group_sizes) == num_group_sizes
    ), f"Expecting {num_group_sizes} distinct group sizes (it's {len(multiple_group_sizes)})."
    return multiple_group_sizes


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
