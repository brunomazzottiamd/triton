# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import logging
import typing
from typing import Any, Callable

# PyTorch
import torch
from torch import Tensor

# Triton
import triton
import triton.language as tl

# Types module
from dtypes import DTYPE

# Common module
from common import TRANS_OUT, is_power_of_2, check_input_device_dtype
from gmm_common import get_gmm_shape, get_gmm_output, get_gmm_transposition
from triton_common import full_tuning_space

# GMM kernel
from triton_gmm_kernel import triton_gmm_kernel_core

# Tuning database
from best_config import pick_best_gmm_config, unique_triton_gmm_configs


# Triton GMM implementation.
# ------------------------------------------------------------------------------


def gmm_heuristics() -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {
        "K_DIVISIBLE_BY_BLOCK_SIZE_K": lambda META: META["K"] % META["BLOCK_SIZE_K"]
        == 0,
    }


def gmm_autotune_configs(use_full_tuning_space: bool = False) -> list[triton.Config]:
    return full_tuning_space() if use_full_tuning_space else unique_triton_gmm_configs()


@triton.heuristics(gmm_heuristics())
@triton.jit
@typing.no_type_check
def triton_gmm_kernel(
    # Tensor pointers:
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    # Tensor shapes:
    M: int,
    K: int,
    N: int,
    G: int,
    # Tensor strides:
    stride_lhs_m: int,
    stride_lhs_k: int,
    stride_rhs_g: int,
    stride_rhs_k: int,
    stride_rhs_n: int,
    stride_out_m: int,
    stride_out_n: int,
    # Meta-parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    # fmt: off
    triton_gmm_kernel_core(
        # Tensor pointers:
        lhs_ptr, rhs_ptr, group_sizes_ptr, out_ptr,
        # Tensor shapes:
        M, K, N, G,
        # Tensor strides:
        stride_lhs_m, stride_lhs_k,
        stride_rhs_g, stride_rhs_k, stride_rhs_n,
        stride_out_m, stride_out_n,
        # Meta-parameters:
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        K_DIVISIBLE_BY_BLOCK_SIZE_K=K_DIVISIBLE_BY_BLOCK_SIZE_K,
        GROUP_SIZE=GROUP_SIZE,
        GRID_DIM=GRID_DIM,
    )
    # fmt: on


@triton.autotune(configs=gmm_autotune_configs(), key=["M", "K", "N", "G"])
@triton.heuristics(gmm_heuristics())
@triton.jit
@typing.no_type_check
def triton_gmm_kernel_autotuned(
    # Tensor pointers:
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    # Tensor shapes:
    M: int,
    K: int,
    N: int,
    G: int,
    # Tensor strides:
    stride_lhs_m: int,
    stride_lhs_k: int,
    stride_rhs_g: int,
    stride_rhs_k: int,
    stride_rhs_n: int,
    stride_out_m: int,
    stride_out_n: int,
    # Meta-parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    # fmt: off
    triton_gmm_kernel_core(
        # Tensor pointers:
        lhs_ptr, rhs_ptr, group_sizes_ptr, out_ptr,
        # Tensor shapes:
        M, K, N, G,
        # Tensor strides:
        stride_lhs_m, stride_lhs_k,
        stride_rhs_g, stride_rhs_k, stride_rhs_n,
        stride_out_m, stride_out_n,
        # Meta-parameters:
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        K_DIVISIBLE_BY_BLOCK_SIZE_K=K_DIVISIBLE_BY_BLOCK_SIZE_K,
        GROUP_SIZE=GROUP_SIZE,
        GRID_DIM=GRID_DIM,
    )
    # fmt: on


def compute_grid(
    N: int,
    block_size_m: int,
    block_size_n: int,
    group_sizes: Tensor,
    grid_dim: int,
) -> tuple[int]:
    assert N > 0, f"N must be positive, it's {N}."
    assert is_power_of_2(
        block_size_m
    ), f"M-dimension tile size must be a power of 2 (it's {block_size_m})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    assert torch.all(group_sizes >= 0).item(), "All group_sizes must be non-negative."
    assert grid_dim > 0, f"Grid dimension must be positive (it's {grid_dim})."
    num_m_tiles = (group_sizes + block_size_m - 1) // block_size_m
    assert torch.all(num_m_tiles >= 0).item(), "All num_m_tiles must be non-negative."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = torch.sum(num_m_tiles * num_n_tiles).item()
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = int(min(grid_dim, num_tiles))
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)


def triton_gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    trans_out: bool = TRANS_OUT,
    existing_out: Tensor | None = None,
    autotune: bool = False,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)

    M, K, N, G = get_gmm_shape(lhs, rhs, group_sizes)

    out = get_gmm_output(
        M,
        N,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        trans=trans_out,
        existing_out=existing_out,
    )

    if not autotune:
        trans_lhs, trans_rhs, trans_out, _, _, _ = get_gmm_transposition(lhs, rhs, out)

        best_config = pick_best_gmm_config(
            M,
            K,
            N,
            G,
            group_sizes=group_sizes,
            input_type=lhs.dtype,
            output_type=out.dtype,
            trans_lhs=trans_lhs,
            trans_rhs=trans_rhs,
            trans_out=trans_out,
        )

        assert best_config.grid_dim is not None, "Unexpected absent grid dimension."
        grid = compute_grid(
            N,
            best_config.block_size_m,
            best_config.block_size_n,
            group_sizes,
            best_config.grid_dim,
        )

        # fmt: off
        triton_gmm_kernel[grid](
            # Tensor pointers:
            lhs, rhs, group_sizes, out,
            # Tensor shapes:
            M, K, N, G,
            # Tensor strides:
            *lhs.stride(), *rhs.stride(), *out.stride(),
            # Meta-parameters:
            BLOCK_SIZE_M=best_config.block_size_m,
            BLOCK_SIZE_K=best_config.block_size_k,
            BLOCK_SIZE_N=best_config.block_size_n,
            GROUP_SIZE=best_config.group_size,
            GRID_DIM=best_config.grid_dim,
        )
        # fmt: on

    else:
        logging.debug("Running autotuned GMM kernel.")

        autotuned_grid = lambda META: compute_grid(
            N,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_N"],
            group_sizes,
            META["GRID_DIM"],
        )

        # fmt: off
        triton_gmm_kernel_autotuned[autotuned_grid](
            # Tensor pointers:
            lhs, rhs, group_sizes, out,
            # Tensor shapes:
            M, K, N, G,
            # Tensor strides:
            *lhs.stride(), *rhs.stride(), *out.stride(),
        )
        # fmt: on

    return out
