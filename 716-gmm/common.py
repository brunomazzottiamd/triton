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


# Supported data types, as strings.
SUPPORTED_DTYPES_STR: set[str] = {"fp16", "bf16", "fp32"}


# Convert string data type to PyTorch data type.
def dtype_from_str(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.strip().lower()
    dtype_str = dtype_str[1:] if dtype_str[0] in {"i", "o"} else dtype_str
    assert (
        dtype_str in SUPPORTED_DTYPES_STR
    ), "String data type isn't in set of supported string data types."
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        dtype_str
    ]


# Supported data types, as PyTorch types.
SUPPORTED_DTYPES: set[torch.dtype] = {
    dtype_from_str(dtype_str) for dtype_str in SUPPORTED_DTYPES_STR
}


def str_from_dtype(dtype: torch.dtype) -> str:
    assert (
        dtype in SUPPORTED_DTYPES
    ), "PyTorch data type isn't in set of supported PyTorch data types."
    return {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]


# Default data type, as string.
DTYPE_STR: str = "bf16"
assert (
    DTYPE_STR in SUPPORTED_DTYPES_STR
), "Default string data type isn't in set of supported string data types."
# Default data type, as PyTorch type.
DTYPE: torch.dtype = dtype_from_str(DTYPE_STR)


# Default RNG seed for input generation.
RNG_SEED: int = 0


# Probabilities for generating random group sizes.
UNUSED_TOKENS_PROB: float = 0.0
UNUSED_EXPERTS_PROB: float = 0.1


# Default number of group sizes to use when benchmarking and launching the kernel for profiling.
NUM_GROUP_SIZES: int = 1


# TODO: Figure out a sensible tiling default.
TILING: tuple[int, int, int] = (64, 64, 64)


# Default transposition.
TRANS_LHS: bool = False
TRANS_RHS: bool = True
TRANS_OUT: bool = False


# Tensor creation functions.
# ------------------------------------------------------------------------------


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


def gen_input(
    M: int,
    K: int,
    N: int,
    G: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
    trans_lhs: bool = TRANS_LHS,
    trans_rhs: bool = TRANS_RHS,
    rng_seed: int | None = RNG_SEED,
) -> tuple[Tensor, Tensor, Tensor]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    if trans_lhs:
        lhs = torch.randn((K, M), dtype=torch.float32, device=device).T
    else:
        lhs = torch.randn((M, K), dtype=torch.float32, device=device)
    lhs = lhs.to(preferred_element_type)

    if trans_rhs:
        rhs = torch.randn((G, N, K), dtype=torch.float32, device=device).permute(
            0, 2, 1
        )
    else:
        rhs = torch.randn((G, K, N), dtype=torch.float32, device=device)
    rhs = rhs.to(preferred_element_type)

    group_sizes = gen_group_sizes(M, G, device=device, rng_seed=None)

    return lhs, rhs, group_sizes


def gen_output(
    M: int,
    N: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
    trans: bool = TRANS_OUT,
) -> Tensor:
    assert M > 0, f"Number of out rows M must be positive (M = {M})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."

    if trans:
        out = torch.zeros((N, M), dtype=preferred_element_type, device=device).T
    else:
        out = torch.zeros((M, N), dtype=preferred_element_type, device=device)

    return out


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


def get_shape_from_input(
    lhs: Tensor, rhs: Tensor, group_sizes: Tensor
) -> tuple[int, int, int, int]:
    assert lhs.dim() == 2, f"lhs must have 2 dimensions (it's {lhs.dim()})."
    assert rhs.dim() == 3, f"rhs must have 3 dimensions (it's {rhs.dim()})."
    assert (
        group_sizes.dim() == 1
    ), f"group_sizes must have 1 dimension (it's {group_sizes.dim()})."

    M, lhs_k = lhs.shape
    rhs_g, rhs_k, N = rhs.shape
    group_sizes_g = group_sizes.shape[0]

    assert (
        lhs_k == rhs_k
    ), f"K dimension of lhs and rhs don't match (lhs = {lhs_k}, rhs = {rhs_k})."
    K = lhs_k
    assert (
        rhs_g == group_sizes_g
    ), f"G dimension of rhs and group_sizes don't match (rhs = {rhs_g}, group_sizes = {group_sizes_g})."
    G = rhs_g

    assert M > 0, f"M must be positive, it's {M}."
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}"
    assert G > 0, f"G must be positive, it's {G}"

    return M, K, N, G


def is_power_of_2(x: int) -> bool:
    return (x > 0) and (x & (x - 1) == 0)


def get_tiling(
    M: int, K: int, N: int, tiling: tuple[int, int, int]
) -> tuple[int, int, int]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert len(tiling) == 3, f"tiling must have 3 dimensions (it's = {len(tiling)})."

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


def get_output(
    M: int,
    N: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
) -> Tensor:
    assert M > 0, f"Number of out rows M must be positive (M = {M})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."

    if existing_out is not None:
        assert (
            existing_out.device == device
        ), f"Existing output device and provided device don't match (existing = {existing_out.device}, provided = {device})."
        assert (
            existing_out.dtype == preferred_element_type
        ), f"Existing output type and preferred output type don't match (existing = {existing_out.dtype}, preferred = {preferred_element_type})."
        assert existing_out.shape == (
            M,
            N,
        ), f"Existing output shape and GMM shape don't match (existing = {tuple(existing_out.shape)}, provided = {(M, N)})."
        return existing_out

    return gen_output(
        M, N, device=device, preferred_element_type=preferred_element_type
    )
