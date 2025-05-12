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
    get_tiling,
    get_output,
)

# GMM kernel
from triton_gmm_kernel import triton_gmm_kernel_core


# Triton GMM implementation.
# ------------------------------------------------------------------------------


def heuristics() -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {
        "K_DIVISIBLE_BY_BLOCK_SIZE_K": lambda META: META["K"] % META["BLOCK_SIZE_K"]
        == 0,
    }


def autotune_configs(full_tuning_space: bool = False) -> list[triton.Config]:
    if not full_tuning_space:
        # fmt: off
        return [
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 32, "BLOCK_SIZE_N": 256}),
            triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 32, "BLOCK_SIZE_N": 128}),
            triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 32, "BLOCK_SIZE_N": 256}),
        ]
        # fmt: on
    block_sizes = [32, 64, 128, 256]
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_K": block_size_k,
                "BLOCK_SIZE_N": block_size_n,
            }
        )
        for block_size_m, block_size_k, block_size_n in itertools.product(
            block_sizes, block_sizes, block_sizes
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
    # Tensor strides (part 1):
    stride_rhs_n: int,  # tl.constexpr,
    # Tensor shapes:
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    # Tensor strides:
    stride_lhs_m: tl.constexpr,
    stride_lhs_k: tl.constexpr,
    stride_rhs_g: tl.constexpr,
    stride_rhs_k: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    # Meta-parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr,
):
    triton_gmm_kernel_core(
        # Tensor pointers:
        lhs_ptr,
        rhs_ptr,
        group_sizes_ptr,
        out_ptr,
        # Tensor strides (part 1):
        stride_rhs_n,
        # Tensor shapes:
        M=M,
        K=K,
        N=N,
        G=G,
        # Tensor strides (part 2):
        stride_lhs_m=stride_lhs_m,
        stride_lhs_k=stride_lhs_k,
        stride_rhs_g=stride_rhs_g,
        stride_rhs_k=stride_rhs_k,
        stride_out_m=stride_out_m,
        stride_out_n=stride_out_n,
        # Meta-parameters:
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        K_DIVISIBLE_BY_BLOCK_SIZE_K=K_DIVISIBLE_BY_BLOCK_SIZE_K,
    )


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
    # Tensor strides (part 1):
    stride_rhs_n: int,  # tl.constexpr,
    # Tensor shapes:
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    # Tensor strides (part 2):
    stride_lhs_m: tl.constexpr,
    stride_lhs_k: tl.constexpr,
    stride_rhs_g: tl.constexpr,
    stride_rhs_k: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    # Meta-parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr,
):
    triton_gmm_kernel_core(
        # Tensor pointers:
        lhs_ptr,
        rhs_ptr,
        group_sizes_ptr,
        out_ptr,
        # Tensor strides (part 1):
        stride_rhs_n,
        # Tensor shapes:
        M=M,
        K=K,
        N=N,
        G=G,
        # Tensor strides (part 2):
        stride_lhs_m=stride_lhs_m,
        stride_lhs_k=stride_lhs_k,
        stride_rhs_g=stride_rhs_g,
        stride_rhs_k=stride_rhs_k,
        stride_out_m=stride_out_m,
        stride_out_n=stride_out_n,
        # Meta-parameters:
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        K_DIVISIBLE_BY_BLOCK_SIZE_K=K_DIVISIBLE_BY_BLOCK_SIZE_K,
    )


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
    tiling: tuple[int, int, int] | None = None,
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

    if tiling is not None:
        block_size_m, block_size_k, block_size_n = get_tiling(M, K, N, tiling)
        logging.debug(
            "Running kernel with tiling (BLOCK_SIZE_M = %d, BLOCK_SIZE_K = %d, BLOCK_SIZE_N = %d).",
            block_size_m,
            block_size_k,
            block_size_n,
        )
        grid = compute_grid(N, block_size_m, block_size_n, group_sizes)
        triton_gmm_kernel[grid](
            # Tensor pointers:
            lhs,
            rhs,
            group_sizes,
            out,
            # Tensor strides (part 1):
            rhs.stride(2),  # stride_rhs_n
            # Tensor shapes:
            M=M,
            K=K,
            N=N,
            G=G,
            # Tensor strides (part 2):
            stride_lhs_m=lhs.stride(0),
            stride_lhs_k=lhs.stride(1),
            stride_rhs_g=rhs.stride(0),
            stride_rhs_k=rhs.stride(1),
            stride_out_m=out.stride(0),
            stride_out_n=out.stride(1),
            # Meta-parameters:
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_K=block_size_k,
            BLOCK_SIZE_N=block_size_n,
        )
    else:
        logging.debug("Running autotuned kernel.")
        autotuned_grid = lambda META: compute_grid(
            N, META["BLOCK_SIZE_M"], META["BLOCK_SIZE_N"], group_sizes
        )
        triton_autotuned_gmm_kernel[autotuned_grid](
            # Tensor pointers:
            lhs,
            rhs,
            group_sizes,
            out,
            # Tensor strides (part 1):
            rhs.stride(2),  # stride_rhs_n
            # Tensor shapes:
            M=M,
            K=K,
            N=N,
            G=G,
            # Tensor strides (part 2):
            stride_lhs_m=lhs.stride(0),
            stride_lhs_k=lhs.stride(1),
            stride_rhs_g=rhs.stride(0),
            stride_rhs_k=rhs.stride(1),
            stride_out_m=out.stride(0),
            stride_out_n=out.stride(1),
        )

    return out
