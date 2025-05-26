# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import logging
import typing

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
from tgmm_common import get_tgmm_shape, get_tgmm_output, get_tgmm_transposition
from triton_common import full_tuning_space

# GMM kernel
from triton_tgmm_kernel import triton_tgmm_kernel_core

# Tuning database
from best_config import pick_best_tgmm_config, unique_triton_tgmm_configs


# Triton TGMM implementation.
# ------------------------------------------------------------------------------


def tgmm_autotune_configs(use_full_tuning_space: bool = False) -> list[triton.Config]:
    return (
        full_tuning_space() if use_full_tuning_space else unique_triton_tgmm_configs()
    )


@triton.jit
@typing.no_type_check
def triton_tgmm_kernel(
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
    stride_lhs_k: int,
    stride_lhs_m: int,
    stride_rhs_m: int,
    stride_rhs_n: int,
    stride_out_g: int,
    stride_out_k: int,
    stride_out_n: int,
    # Meta-parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    # fmt: off
    triton_tgmm_kernel_core(
        # Tensor pointers:
        lhs_ptr, rhs_ptr, group_sizes_ptr, out_ptr,
        # Tensor shapes:
        M, K, N, G,
        # Tensor strides:
        stride_lhs_k, stride_lhs_m,
        stride_rhs_m, stride_rhs_n,
        stride_out_g, stride_out_k, stride_out_n,
        # Meta-parameters:
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE=GROUP_SIZE,
        GRID_DIM=GRID_DIM,
    )
    # fmt: on


@triton.autotune(configs=tgmm_autotune_configs(), key=["M", "K", "N", "G"])
@triton.jit
@typing.no_type_check
def triton_autotuned_tgmm_kernel(
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
    stride_lhs_k: int,
    stride_lhs_m: int,
    stride_rhs_m: int,
    stride_rhs_n: int,
    stride_out_g: int,
    stride_out_k: int,
    stride_out_n: int,
    # Meta-parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    # fmt: off
    triton_tgmm_kernel_core(
        # Tensor pointers:
        lhs_ptr, rhs_ptr, group_sizes_ptr, out_ptr,
        # Tensor shapes:
        M, K, N, G,
        # Tensor strides:
        stride_lhs_k, stride_lhs_m,
        stride_rhs_m, stride_rhs_n,
        stride_out_g, stride_out_k, stride_out_n,
        # Meta-parameters:
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE=GROUP_SIZE,
        GRID_DIM=GRID_DIM,
    )
    # fmt: on


def compute_grid(
    K: int, N: int, G: int, block_size_k: int, block_size_n: int, grid_dim: int
) -> tuple[int]:
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}."
    assert G > 0, f"G must be positive, it's {G}."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    assert grid_dim > 0, f"Grid dimension must be positive (it's {grid_dim})."
    num_k_tiles = triton.cdiv(K, block_size_k)
    assert num_k_tiles > 0, f"num_k_tiles must be positive, it's {num_k_tiles}."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = G * num_k_tiles * num_n_tiles
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = min(grid_dim, num_tiles)
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)


def triton_tgmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    trans_out: bool = TRANS_OUT,
    existing_out: Tensor | None = None,
    autotune: bool = False,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)

    M, K, N, G = get_tgmm_shape(lhs, rhs, group_sizes)

    out = get_tgmm_output(
        K,
        N,
        G,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        trans=trans_out,
        existing_out=existing_out,
    )

    if not autotune:
        trans_lhs, trans_rhs, trans_out, _, _, _ = get_tgmm_transposition(lhs, rhs, out)

        best_config = pick_best_tgmm_config(
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

        grid = compute_grid(
            K,
            N,
            G,
            best_config.block_size_k,
            best_config.block_size_n,
            best_config.grid_dim,
        )

        # fmt: off
        triton_tgmm_kernel[grid](
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
        logging.debug("Running autotuned TGMM kernel.")

        autotuned_grid = lambda META: compute_grid(
            K, N, G, META["BLOCK_SIZE_K"], META["BLOCK_SIZE_N"], META["GRID_DIM"]
        )

        # fmt: off
        triton_autotuned_tgmm_kernel[autotuned_grid](
            # Tensor pointers:
            lhs, rhs, group_sizes, out,
            # Tensor shapes:
            M, K, N, G,
            # Tensor strides:
            *lhs.stride(), *rhs.stride(), *out.stride(),
        )
        # fmt: on

    return out
