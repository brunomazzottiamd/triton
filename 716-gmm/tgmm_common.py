# -*- coding: utf-8 -*-


# TGMM problem description:
# * Input tensors:
#   * lhs is (K, M) bf16
#   * rhs is (M, N) bf16
#   * group_sizes is (G,) int32
# * Output tensors:
#   * out is (G, K, N) bf16


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
    TRANS_LHS,
    TRANS_RHS,
    TRANS_OUT,
)

# Group sizes module
from group_sizes import (
    gen_uniform_group_sizes,
    gen_group_sizes,
    gen_multiple_group_sizes,
)


# Tensor creation functions.
# ------------------------------------------------------------------------------


def gen_tgmm_input(
    M: int,
    K: int,
    N: int,
    G: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
    trans_lhs: bool = TRANS_LHS,
    trans_rhs: bool = TRANS_RHS,
    rng_seed: int | None = RNG_SEED,
    unif_group_sizes: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    assert K > 0, f"Number of lhs rows K must be positive (M = {K})."
    assert M > 0, f"Number of lhs columns / rhs rows M must be positive (K = {M})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    if trans_lhs:
        lhs = torch.randn((M, K), dtype=torch.float32, device=device).T
    else:
        lhs = torch.randn((K, M), dtype=torch.float32, device=device)
    lhs = lhs.to(preferred_element_type)

    if trans_rhs:
        rhs = torch.randn((N, M), dtype=torch.float32, device=device).T
    else:
        rhs = torch.randn((M, N), dtype=torch.float32, device=device)
    rhs = rhs.to(preferred_element_type)

    group_sizes = (
        gen_uniform_group_sizes(M, G, device=device)
        if unif_group_sizes
        else gen_group_sizes(M, G, device=device, rng_seed=None)
    )

    return lhs, rhs, group_sizes


def gen_tgmm_output(
    K: int,
    N: int,
    G: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
    trans: bool = TRANS_OUT,
) -> Tensor:
    assert K > 0, f"Number of out rows K must be positive (K = {K})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    if trans:
        out = torch.empty(
            (G, N, K), dtype=preferred_element_type, device=device
        ).permute(0, 2, 1)
    else:
        out = torch.empty((G, K, N), dtype=preferred_element_type, device=device)

    return out


def gen_tgmm_tensors(
    M: int,
    K: int,
    N: int,
    G: int,
    num_group_sizes: int,
    device: torch.device | str = DEVICE,
    input_type: torch.dtype = DTYPE,
    output_type: torch.dtype = DTYPE,
    trans_lhs: bool = TRANS_LHS,
    trans_rhs: bool = TRANS_RHS,
    trans_out: bool = TRANS_OUT,
    rng_seed: int | None = RNG_SEED,
    unif_group_sizes: bool = False,
) -> tuple[Tensor, Tensor, list[Tensor], Tensor]:
    lhs, rhs, group_sizes_0 = gen_tgmm_input(
        M,
        K,
        N,
        G,
        device=device,
        preferred_element_type=input_type,
        trans_lhs=trans_lhs,
        trans_rhs=trans_rhs,
        rng_seed=rng_seed,
        unif_group_sizes=unif_group_sizes,
    )
    multiple_group_sizes = gen_multiple_group_sizes(
        num_group_sizes, M, G, device=device, rng_seed=None, group_sizes_0=group_sizes_0
    )
    out = gen_tgmm_output(
        K, N, G, device=device, preferred_element_type=output_type, trans=trans_out
    )
    return lhs, rhs, multiple_group_sizes, out


# Functions to extract information from generated tensors.
# ------------------------------------------------------------------------------


def get_tgmm_shape(
    lhs: Tensor, rhs: Tensor, group_sizes: Tensor
) -> tuple[int, int, int, int]:
    assert lhs.dim() == 2, f"lhs must have 2 dimensions (it's {lhs.dim()})."
    assert rhs.dim() == 2, f"rhs must have 2 dimensions (it's {rhs.dim()})."
    assert (
        group_sizes.dim() == 1
    ), f"group_sizes must have 1 dimension (it's {group_sizes.dim()})."

    K, lhs_m = lhs.shape
    rhs_m, N = rhs.shape
    G = group_sizes.shape[0]

    assert (
        lhs_m == rhs_m
    ), f"M dimension of lhs and rhs don't match (lhs = {lhs_m}, rhs = {rhs_m})."
    M = lhs_m

    assert M > 0, f"M must be positive, it's {M}."
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}"
    assert G > 0, f"G must be positive, it's {G}"

    return M, K, N, G


def get_tgmm_output(
    K: int,
    N: int,
    G: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
    trans: bool = TRANS_OUT,
    existing_out: Tensor | None = None,
) -> Tensor:
    assert K > 0, f"Number of out rows K must be positive (K = {K})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    if existing_out is not None:
        assert (
            existing_out.device == device
        ), f"Existing output device and provided device don't match (existing = {existing_out.device}, provided = {device})."
        assert (
            existing_out.dtype == preferred_element_type
        ), f"Existing output type and preferred output type don't match (existing = {existing_out.dtype}, preferred = {preferred_element_type})."
        assert existing_out.shape == (
            G,
            K,
            N,
        ), f"Existing output shape and GMM shape don't match (existing = {tuple(existing_out.shape)}, provided = {(G, K, N)})."
        is_out_row_major = existing_out.stride() == (K * N, N, 1)
        is_out_col_major = existing_out.stride() == (K * N, 1, K)
        assert (
            is_out_row_major != is_out_col_major
        ), "Existing output must be row-major or column-major."
        assert (
            is_out_col_major == trans
        ), f"Existing output transposition and provided transposition don't match (existing = {is_out_col_major}, provided = {trans})."
        return existing_out

    return gen_tgmm_output(
        K,
        N,
        G,
        device=device,
        preferred_element_type=preferred_element_type,
        trans=trans,
    )


def get_tgmm_transposition(
    lhs: Tensor, rhs: Tensor, out: Tensor
) -> tuple[bool, bool, bool, int, int, int]:
    assert lhs.dim() == 2, f"lhs must have 2 dimensions (it's {lhs.dim()})."
    assert rhs.dim() == 2, f"rhs must have 2 dimensions (it's {rhs.dim()})."
    assert out.dim() == 3, f"out must have 3 dimensions (it's {out.dim()})."

    lhs_k, lhs_m = lhs.shape
    rhs_m, rhs_n = rhs.shape
    G, out_k, out_n = out.shape

    assert (
        lhs_m == rhs_m
    ), f"M dimension of lhs and rhs don't match (lhs = {lhs_m}, rhs = {rhs_m})."
    M = lhs_m
    assert (
        lhs_k == out_k
    ), f"K dimension of lhs and out don't match (lhs = {lhs_k}, rhs = {out_k})."
    K = lhs_k
    assert (
        rhs_n == out_n
    ), f"N dimension of rhs and out don't match (lhs = {rhs_n}, rhs = {out_n})."
    N = rhs_n

    assert M > 0, f"M must be positive, it's {M}."
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}"
    assert G > 0, f"G must be positive, it's {G}"

    is_lhs_row_major = lhs.stride() == (M, 1)
    is_lhs_col_major = lhs.stride() == (1, K)
    assert (
        is_lhs_row_major != is_lhs_col_major
    ), "lhs must be row-major or column-major."
    is_rhs_row_major = rhs.stride() == (N, 1)
    is_rhs_col_major = rhs.stride() == (1, M)
    assert (
        is_rhs_row_major != is_rhs_col_major
    ), "rhs must be row-major or column-major."
    is_out_row_major = out.stride() == (K * N, N, 1)
    is_out_col_major = out.stride() == (K * N, 1, K)
    assert (
        is_out_row_major != is_out_col_major
    ), "out must be row-major or column-major."

    # Get leading dimension according to transposition configuration.
    ld_lhs = M if is_lhs_row_major else K
    ld_rhs = N if is_rhs_row_major else M
    ld_out = N if is_out_row_major else K

    return is_lhs_col_major, is_rhs_col_major, is_out_col_major, ld_lhs, ld_rhs, ld_out
