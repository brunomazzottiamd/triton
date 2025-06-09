# -*- coding: utf-8 -*-


# GMM problem description:
# * Input tensors:
#   * lhs is (M, K) bf16
#   * rhs is (G, K, N) bf16
#   * group_sizes is (G,) int32
# * Output tensors:
#   * out is (M, N) bf16


# Imports.
# ------------------------------------------------------------------------------

# PyTorch
import torch
from torch import Tensor

# Types module
from dtypes import DTYPE

# Common module
from common import (
    DEVICE,
    RNG_SEED,
)

# Group sizes module
from group_sizes import (
    gen_uniform_group_sizes,
    gen_group_sizes,
    gen_multiple_group_sizes,
)


# Tensor creation functions.
# ------------------------------------------------------------------------------


def gen_gmm_input(
    M: int,
    K: int,
    N: int,
    G: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
    rng_seed: int | None = RNG_SEED,
    unif_group_sizes: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    lhs = torch.randn((M, K), dtype=torch.float32, device=device)
    lhs = lhs.to(preferred_element_type)

    rhs = torch.randn((G, K, N), dtype=torch.float32, device=device)
    rhs = rhs.to(preferred_element_type)

    group_sizes = (
        gen_uniform_group_sizes(M, G, device=device)
        if unif_group_sizes
        else gen_group_sizes(M, G, device=device, rng_seed=None)
    )

    return lhs, rhs, group_sizes


def gen_gmm_output(
    M: int,
    N: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
) -> Tensor:
    assert M > 0, f"Number of out rows M must be positive (M = {M})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."

    out = torch.empty((M, N), dtype=preferred_element_type, device=device)

    return out


def gen_gmm_tensors(
    M: int,
    K: int,
    N: int,
    G: int,
    num_group_sizes: int,
    device: torch.device | str = DEVICE,
    input_type: torch.dtype = DTYPE,
    output_type: torch.dtype = DTYPE,
    rng_seed: int | None = RNG_SEED,
    unif_group_sizes: bool = False,
) -> tuple[Tensor, Tensor, list[Tensor], Tensor]:
    lhs, rhs, group_sizes_0 = gen_gmm_input(
        M,
        K,
        N,
        G,
        device=device,
        preferred_element_type=input_type,
        rng_seed=rng_seed,
        unif_group_sizes=unif_group_sizes,
    )
    multiple_group_sizes = gen_multiple_group_sizes(
        num_group_sizes, M, G, device=device, rng_seed=None, group_sizes_0=group_sizes_0
    )
    out = gen_gmm_output(M, N, device=device, preferred_element_type=output_type)
    return lhs, rhs, multiple_group_sizes, out


# Functions to extract information from generated tensors.
# ------------------------------------------------------------------------------


def get_gmm_shape(
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


def get_gmm_output(
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

    return gen_gmm_output(
        M,
        N,
        device=device,
        preferred_element_type=preferred_element_type,
    )
