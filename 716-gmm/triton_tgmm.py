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
from common import is_power_of_2, check_input_device_dtype
from tgmm_common import get_tgmm_shape, get_tgmm_output
from triton_common import full_tuning_space

# GMM kernel
from triton_tgmm_kernel import (
    triton_tgmm_persistent_kernel_core,
    triton_tgmm_non_persistent_kernel_core,
)

# Tuning database
from best_config import (
    pick_best_persistent_tgmm_config,
    pick_best_non_persistent_tgmm_config,
    unique_triton_persistent_tgmm_configs,
    unique_triton_non_persistent_tgmm_configs,
)


# Triton persistent TGMM implementation.
# ------------------------------------------------------------------------------


def tgmm_persistent_autotune_configs(
    use_full_tuning_space: bool = False,
) -> list[triton.Config]:
    return (
        full_tuning_space()
        if use_full_tuning_space
        else unique_triton_persistent_tgmm_configs()
    )


@triton.jit
@typing.no_type_check
def triton_tgmm_persistent_kernel(
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
    # Meta-parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    # fmt: off
    triton_tgmm_persistent_kernel_core(
        # Tensor pointers:
        lhs_ptr, rhs_ptr, group_sizes_ptr, out_ptr,
        # Tensor shapes:
        M, K, N, G,
        # Meta-parameters:
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE=GROUP_SIZE,
        GRID_DIM=GRID_DIM,
    )
    # fmt: on


@triton.autotune(configs=tgmm_persistent_autotune_configs(), key=["M", "K", "N", "G"])
@triton.jit
@typing.no_type_check
def triton_tgmm_persistent_autotuned_kernel(
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
    # Meta-parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    # fmt: off
    triton_tgmm_persistent_kernel_core(
        # Tensor pointers:
        lhs_ptr, rhs_ptr, group_sizes_ptr, out_ptr,
        # Tensor shapes:
        M, K, N, G,
        # Meta-parameters:
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE=GROUP_SIZE,
        GRID_DIM=GRID_DIM,
    )
    # fmt: on


def compute_persistent_grid(
    K: int,
    N: int,
    G: int,
    block_size_k: int,
    block_size_n: int,
    grid_dim: int,
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


def triton_persistent_tgmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
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
        existing_out=existing_out,
    )

    if not autotune:
        best_config = pick_best_persistent_tgmm_config(
            M,
            K,
            N,
            G,
            group_sizes=group_sizes,
            input_type=lhs.dtype,
            output_type=out.dtype,
        )

        assert best_config.grid_dim is not None, "Unexpected absent grid dimension."
        grid = compute_persistent_grid(
            K,
            N,
            G,
            best_config.block_size_k,
            best_config.block_size_n,
            best_config.grid_dim,
        )

        # fmt: off
        triton_tgmm_persistent_kernel[grid](
            # Tensor pointers:
            lhs, rhs, group_sizes, out,
            # Tensor shapes:
            M, K, N, G,
            # Meta-parameters:
            BLOCK_SIZE_M=best_config.block_size_m,
            BLOCK_SIZE_K=best_config.block_size_k,
            BLOCK_SIZE_N=best_config.block_size_n,
            GROUP_SIZE=best_config.group_size,
            GRID_DIM=best_config.grid_dim,
        )
        # fmt: on

    else:
        logging.debug("Running autotuned persistent TGMM kernel.")

        autotuned_grid = lambda META: compute_persistent_grid(
            K,
            N,
            G,
            META["BLOCK_SIZE_K"],
            META["BLOCK_SIZE_N"],
            META["GRID_DIM"],
        )

        # fmt: off
        triton_tgmm_persistent_autotuned_kernel[autotuned_grid](
            # Tensor pointers:
            lhs, rhs, group_sizes, out,
            # Tensor shapes:
            M, K, N, G,
        )
        # fmt: on

    return out


# Triton non-persistent TGMM implementation.
# ------------------------------------------------------------------------------


def tgmm_non_persistent_heuristics() -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {"BLOCK_SIZE_G": lambda META: triton.next_power_of_2(META["G"])}


def tgmm_non_persistent_autotune_configs(
    use_full_tuning_space: bool = False,
) -> list[triton.Config]:
    return (
        full_tuning_space(add_grid_dim=False)
        if use_full_tuning_space
        else unique_triton_non_persistent_tgmm_configs()
    )


@triton.heuristics(tgmm_non_persistent_heuristics())
@triton.jit
@typing.no_type_check
def triton_tgmm_non_persistent_kernel(
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
    # Meta-parameters:
    BLOCK_SIZE_G: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # fmt: off
    triton_tgmm_non_persistent_kernel_core(
        # Tensor pointers:
        lhs_ptr, rhs_ptr, group_sizes_ptr, out_ptr,
        # Tensor shapes:
        M, K, N, G,
        # Meta-parameters:
        BLOCK_SIZE_G=BLOCK_SIZE_G,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE=GROUP_SIZE,
    )
    # fmt: on


@triton.autotune(
    configs=tgmm_non_persistent_autotune_configs(), key=["M", "K", "N", "G"]
)
@triton.heuristics(tgmm_non_persistent_heuristics())
@triton.jit
@typing.no_type_check
def triton_tgmm_non_persistent_autotuned_kernel(
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
    # Meta-parameters:
    BLOCK_SIZE_G: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # fmt: off
    triton_tgmm_non_persistent_kernel_core(
        # Tensor pointers:
        lhs_ptr, rhs_ptr, group_sizes_ptr, out_ptr,
        # Tensor shapes:
        M, K, N, G,
        # Meta-parameters:
        BLOCK_SIZE_G=BLOCK_SIZE_G,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE=GROUP_SIZE,
    )
    # fmt: on


def compute_non_persistent_grid(
    K: int,
    N: int,
    G: int,
    block_size_k: int,
    block_size_n: int,
) -> tuple[int, int]:
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}."
    assert G > 0, f"G must be positive, it's {G}."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    num_k_tiles = triton.cdiv(K, block_size_k)
    assert num_k_tiles > 0, f"num_k_tiles must be positive, it's {num_k_tiles}."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles_per_mm = num_k_tiles * num_n_tiles
    assert (
        num_tiles_per_mm > 0
    ), f"num_tiles_per_mm must be positive, it's {num_tiles_per_mm}."
    return (G, num_tiles_per_mm)


def triton_non_persistent_tgmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
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
        existing_out=existing_out,
    )

    if not autotune:
        best_config = pick_best_non_persistent_tgmm_config(
            M,
            K,
            N,
            G,
            group_sizes=group_sizes,
            input_type=lhs.dtype,
            output_type=out.dtype,
        )

        assert best_config.grid_dim is None, "Unexpected existing grid dimension."
        grid = compute_non_persistent_grid(
            K,
            N,
            G,
            best_config.block_size_k,
            best_config.block_size_n,
        )

        # fmt: off
        triton_tgmm_non_persistent_kernel[grid](
            # Tensor pointers:
            lhs, rhs, group_sizes, out,
            # Tensor shapes:
            M, K, N, G,
            # Meta-parameters:
            BLOCK_SIZE_M=best_config.block_size_m,
            BLOCK_SIZE_K=best_config.block_size_k,
            BLOCK_SIZE_N=best_config.block_size_n,
            GROUP_SIZE=best_config.group_size,
        )
        # fmt: on

    else:
        logging.debug("Running autotuned non-persistent TGMM kernel.")

        autotuned_grid = lambda META: compute_non_persistent_grid(
            K,
            N,
            G,
            META["BLOCK_SIZE_K"],
            META["BLOCK_SIZE_N"],
        )

        # fmt: off
        triton_tgmm_non_persistent_autotuned_kernel[autotuned_grid](
            # Tensor pointers:
            lhs, rhs, group_sizes, out,
            # Tensor shapes:
            M, K, N, G,
        )
        # fmt: on

    return out
