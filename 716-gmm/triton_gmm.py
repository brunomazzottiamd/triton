# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import itertools
import logging
import typing
from typing import Any, Callable

# PyTorch
import torch
from torch import Tensor

# Triton
import triton
import triton.language as tl

# Common module
from common import (
    DEVICE,
    DTYPE,
    is_power_of_2,
    check_input_device_dtype,
    get_shape_from_input,
    get_output,
    get_transposition,
)

# GMM kernel
from triton_gmm_kernel import triton_gmm_kernel_core

# Tuning database
from best_config import BEST_CONFIGS, pick_best_config


# Triton GMM implementation.
# ------------------------------------------------------------------------------


def heuristics() -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {
        "K_DIVISIBLE_BY_BLOCK_SIZE_K": lambda META: META["K"] % META["BLOCK_SIZE_K"]
        == 0,
    }


def autotune_configs(full_tuning_space: bool = False) -> list[triton.Config]:
    if not full_tuning_space:
        # Grab all distinct configs from tuning database.
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": config.block_size_m,
                    "BLOCK_SIZE_K": config.block_size_k,
                    "BLOCK_SIZE_N": config.block_size_n,
                    "GROUP_SIZE_M": config.group_size_m,
                },
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
            for config in set(BEST_CONFIGS.values())
        ]

    # Generate lots of configs with Cartesian product approach.
    block_sizes = [32, 64, 128, 256]
    # |_ Consider restricting block_size_m_range to [64, 128, 256].
    # |_ Consider restricting block_size_k_range to [32].
    # |_ Consider restricting block_size_n_range to [128, 256].
    group_size_m_range = [1, 2, 4, 8]
    num_warps_range = [2, 4, 8]
    # |_ Consider restricting num_warps_range to [4, 8].
    num_stages_range = [1, 2]
    # waves_per_eu_range = [0, 2, 4, 8]
    # matrix_instr_nonkdim_range = [16, 32]
    # kpack_range = [1, 2]
    # 1536 configurations per shape
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_K": block_size_k,
                "BLOCK_SIZE_N": block_size_n,
                "GROUP_SIZE_M": group_size_m,
                # "waves_per_eu": waves_per_eu,
                # "kpack": kpack,
                # "matrix_instr_nonkdim": matrix_instr_nonkdim,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_size_m, block_size_k, block_size_n, group_size_m, num_warps, num_stages in itertools.product(
            block_sizes,
            block_sizes,
            block_sizes,
            group_size_m_range,
            num_warps_range,
            num_stages_range,
        )
    ]


@triton.heuristics(heuristics())
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
    GROUP_SIZE_M: tl.constexpr,
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
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    # fmt: on


@triton.autotune(configs=autotune_configs(), key=["M", "K", "N", "G"])
@triton.heuristics(heuristics())
@triton.jit
@typing.no_type_check
def triton_autotuned_gmm_kernel(
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
    GROUP_SIZE_M: tl.constexpr,
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
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    # fmt: on


def num_sms(device: torch.device | str = DEVICE) -> int:
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    assert num_sms, f"Number of SMs must be positive (it's {num_sms})."
    return num_sms


def compute_grid(
    N: int,
    block_size_m: int,
    block_size_n: int,
    group_sizes: Tensor,
) -> tuple[int]:
    assert N > 0, f"N must be positive, it's {N}."
    assert is_power_of_2(
        block_size_m
    ), f"M-dimension tile size must be a power of 2 (it's {block_size_m})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    assert torch.all(group_sizes >= 0).item(), "All group_sizes must be non-negative."
    num_m_tiles = (group_sizes + block_size_m - 1) // block_size_m
    assert torch.all(num_m_tiles >= 0).item(), "All num_m_tiles must be non-negative."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = torch.sum(num_m_tiles * num_n_tiles).item()
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = int(min(num_sms(device=group_sizes.device), num_tiles))
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)


def triton_gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
    autotune: bool = False,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)

    M, K, N, G = get_shape_from_input(lhs, rhs, group_sizes)

    out = get_output(
        M,
        N,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        existing_out=existing_out,
    )

    trans_lhs, trans_rhs, trans_out, ld_lhs, ld_rhs, ld_out = get_transposition(
        lhs, rhs, out
    )

    if not autotune:
        best_config = pick_best_config(
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
            N, best_config.block_size_m, best_config.block_size_n, group_sizes
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
            GROUP_SIZE_M=best_config.group_size_m,
        )
        # fmt: on

    else:
        logging.debug("Running autotuned kernel.")

        autotuned_grid = lambda META: compute_grid(
            N, META["BLOCK_SIZE_M"], META["BLOCK_SIZE_N"], group_sizes
        )

        # fmt: off
        triton_autotuned_gmm_kernel[autotuned_grid](
            # Tensor pointers:
            lhs, rhs, group_sizes, out,
            # Tensor shapes:
            M, K, N, G,
            # Tensor strides:
            *lhs.stride(), *rhs.stride(), *out.stride(),
        )
        # fmt: on

    return out
