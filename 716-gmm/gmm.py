# -*- coding: utf-8 -*-


# GMM problem description:
# * Input tensors:
#   * lhs is (M, K) bf16
#   * rhs is (G, K, N) bf16
#   * group_sizes is (G,) int32
# * Output tensors:
#   * out is (M, N) bf16


import random

import torch
from torch import Tensor

import triton
import triton.language as tl

import pytest


DEVICE: str = "cuda"


def random_group_sizes(M: int, G: int, rng_seed: int | None = None) -> list[int]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."
    assert G <= M, f"Cannot split M into more than M groups (M = {M}, G = {G})."

    if rng_seed is not None:
        random.seed(rng_seed)

    # Generate G - 1 sorted cut points between 1 and M - 1.
    cuts = sorted(random.sample(range(1, M), G - 1))
    # Add 0 at the beginning and M at the end, then take differences.
    group_sizes = [b - a for a, b in zip([0] + cuts, cuts + [M])]

    assert len(group_sizes) == G
    assert sum(group_sizes) == M

    return group_sizes


def gen_input(
    M: int, K: int, N: int, G: int, rng_seed: int | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    lhs = torch.randn((M, K), dtype=torch.float32, device=DEVICE)
    lhs = lhs.to(torch.bfloat16)

    rhs = torch.randn((G, K, N), dtype=torch.float32, device=DEVICE)
    rhs = rhs.to(torch.bfloat16)

    group_sizes = random_group_sizes(M, G, rng_seed=rng_seed)
    group_sizes = torch.tensor(group_sizes, dtype=torch.int32, device=DEVICE)

    return lhs, rhs, group_sizes


def gen_output(M: int, N: int) -> Tensor:
    assert M > 0, f"Number of out rows M must be positive (M = {M})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."

    out = torch.empty((M, N), dtype=torch.bfloat16, device=DEVICE)

    return out


def shapes_from_input(
    lhs: Tensor, rhs: Tensor, group_sizes: Tensor
) -> tuple[int, int, int, int]:
    assert lhs.dim() == 2, f"lhs must have 2 dimensions (lhs.dim() = {lhs.dim()})."
    assert rhs.dim() == 3, f"rhs must have 3 dimensions (rhs.dim() = {rhs.dim()})."
    assert (
        group_sizes.dim() == 1
    ), f"group_sizes must have 1 dimension (group_sizes.dim() = {group_sizes.dim()})."

    M, lhs_k = lhs.shape
    rhs_g, rhs_k, N = rhs.shape
    group_sizes_g = group_sizes.shape[0]

    assert (
        lhs_k == rhs_k
    ), f"K dimension of lhs and rhs don't match ({lhs_k} != {rhs_k})."
    K = lhs_k
    assert (
        rhs_g == group_sizes_g
    ), f"G dimension of rhs and group_sizes don't match ({rhs_g} != {group_sizes_g})."
    G = rhs_g

    return M, K, N, G


def torch_gmm(
    lhs: Tensor, rhs: Tensor, group_sizes: Tensor, out: Tensor | None = None
) -> Tensor:
    M, _, N, G = shapes_from_input(lhs, rhs, group_sizes)

    if out is None:
        out = gen_output(M, N)

    offsets = torch.zeros(G + 1, dtype=torch.int32, device=DEVICE)
    torch.cumsum(group_sizes, dim=0, out=offsets[1:])

    for g in range(G):
        start_idx = offsets[g]
        end_idx = offsets[g + 1]
        out[start_idx:end_idx, :] = lhs[start_idx:end_idx, :] @ rhs[g]

    return out


def num_sms() -> int:
    num_sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    assert num_sms, f"Number of SMs must be positive (it's {num_sms})."
    return num_sms


def is_power_of_2(x: int) -> bool:
    return (x > 0) and (x & (x - 1) == 0)


@triton.jit
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
    stride_group_sizes_g: int,
    stride_out_m: int,
    stride_out_n: int,
    # Meta-parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(K > 0)
    tl.assume(N > 0)
    tl.assume(G > 0)

    tl.assume(stride_lhs_m > 0)
    tl.assume(stride_lhs_k > 0)
    tl.assume(stride_rhs_g > 0)
    tl.assume(stride_rhs_k > 0)
    tl.assume(stride_rhs_n > 0)
    tl.assume(stride_group_sizes_g > 0)
    tl.assume(stride_out_m > 0)
    tl.assume(stride_out_n > 0)

    num_programs = tl.num_programs(0)

    tile = tl.program_id(0)  # Current tile.
    last_mm_end = 0  # Tile limit of last MM problem.
    last_lhs_row = 0  # Last row of lhs, each group reads some rows of lhs.

    # N dimension of all MM problems is the same, so the number of tiles in N dimension.
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)

    # Loop through all (m, K, N) MM problems:
    #   (m, K) x (K, N) = (m, N)
    #   sum(m) = M
    for g in range(G):
        # Get m dimension of current MM problem.
        gm = tl.load(group_sizes_ptr + g * stride_group_sizes_g)

        # Number of tiles in M dimension and total tiles of current MM problem.
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_tiles = num_m_tiles * num_n_tiles

        # Loop through tiles of current MM problem.
        while tile >= last_mm_end and tile < last_mm_end + num_tiles:
            # Figure out tile coordinates in current MM problem.
            tile_in_mm = tile - last_mm_end
            tile_m = tile_in_mm // num_n_tiles
            tile_n = tile_in_mm % num_n_tiles

            # Do regular MM:
            offs_lhs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_rhs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            lhs_ptrs = (
                lhs_ptr
                + (last_lhs_row + offs_lhs_m[:, None]) * stride_lhs_m
                + offs_k[None, :] * stride_lhs_k
            )
            rhs_ptrs = (
                rhs_ptr
                + g * stride_rhs_g
                + offs_k[:, None] * stride_rhs_k
                + offs_rhs_n[None, :] * stride_rhs_n
            )
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                lhs = tl.load(
                    lhs_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0
                )
                rhs = tl.load(
                    rhs_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0
                )
                acc += tl.dot(lhs, rhs)
                lhs_ptrs += BLOCK_SIZE_K * stride_lhs_k
                rhs_ptrs += BLOCK_SIZE_K * stride_rhs_k
            acc = acc.to(out_ptr.type.element_ty)
            offs_out_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_out_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            out_ptrs = (
                out_ptr
                + offs_out_m[:, None] * stride_out_m
                + offs_out_n[None, :] * stride_out_n
            )
            tl.store(
                out_ptrs,
                acc,
                mask=(offs_out_m[:, None] < M) & (offs_out_n[None, :] < N),
            )

            # Go to the next tile by advancing number of programs.
            tile += num_programs

        # Get ready to go to the next MM problem.
        last_mm_end += num_tiles
        last_lhs_row += gm


def triton_gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    tiling: tuple[int, int, int] | None = None,
    out: Tensor | None = None,
) -> Tensor:
    M, K, N, G = shapes_from_input(lhs, rhs, group_sizes)

    if tiling is None:
        # TODO: Figure out a sensible tiling default.
        tiling = (64, 64, 64)

    assert (
        len(tiling) == 3
    ), f"tiling must have 3 dimensions (len(tiling) = {len(tiling)})."
    block_size_m, block_size_k, block_size_n = tiling
    assert is_power_of_2(
        block_size_m
    ), f"M-dimension tile size must be a power of 2 (it's {block_size_m})."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."

    if out is None:
        out = gen_output(M, N)

    grid = (num_sms(),)
    triton_gmm_kernel[grid](
        # Tensor pointers:
        lhs,
        rhs,
        group_sizes,
        out,
        # Tensor shapes:
        M,
        K,
        N,
        G,
        # Tensor strides:
        *lhs.stride(),
        *rhs.stride(),
        *group_sizes.stride(),
        *out.stride(),
        # Meta-parameters:
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_N=block_size_n,
    )

    return out


@pytest.mark.parametrize(
    "M, K, N, G",
    [
        (32, 16, 8, 4),  # Test 1
        (512, 4096, 2048, 160),  # Test 2
        (49152, 1408, 2048, 64),  # deepseekv2-16B
        # (3145728, 2048, 1408, 8),  # deepseekv2-16B (IT'S BIG!)
        (393216, 2048, 1408, 64),  # deepseekv2-16B
        (32768, 6144, 16384, 8),  # Mixtral 8x22B proxy model
        (32768, 16384, 6144, 8),  # Mixtral 8x22B proxy model
    ],
)
def test_gmm(M: int, K: int, N: int, G: int):
    lhs, rhs, group_sizes = gen_input(M, K, N, G, rng_seed=0)
    out_torch = torch_gmm(lhs, rhs, group_sizes)
    out_triton = triton_gmm(lhs, rhs, group_sizes)
    torch.testing.assert_close(out_torch, out_triton, atol=5e-3, rtol=1e-2)
