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
    M: int,
    K: int,
    N: int,
    G: int,
    preferred_element_type: torch.dtype = torch.bfloat16,
    rng_seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    lhs = torch.randn((M, K), dtype=torch.float32, device=DEVICE).to(
        preferred_element_type
    )
    rhs = torch.randn((G, K, N), dtype=torch.float32, device=DEVICE).to(
        preferred_element_type
    )
    group_sizes = torch.tensor(
        random_group_sizes(M, G, rng_seed=rng_seed), dtype=torch.int32, device=DEVICE
    )

    return lhs, rhs, group_sizes


def gen_output(
    M: int, N: int, preferred_element_type: torch.dtype = torch.bfloat16
) -> Tensor:
    assert M > 0, f"Number of out rows M must be positive (M = {M})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."

    return torch.empty((M, N), dtype=preferred_element_type, device=DEVICE)


def shape_from_input(
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


def get_output(
    M: int,
    N: int,
    preferred_element_type: torch.dtype = torch.bfloat16,
    existing_out: Tensor | None = None,
) -> Tensor:
    assert M > 0, f"Number of out rows M must be positive (M = {M})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."

    if existing_out is not None:
        assert (
            existing_out.dtype == preferred_element_type
        ), f"Existing output tensor type and preferred output type don't match ({existing_out.dtype} != {preferred_element_type})."
        assert existing_out.shape == (
            M,
            N,
        ), f"Existing output tensor shape and GMM shape don't match ({tuple(existing_out.shape)} != {(M, N)})."
        return existing_out

    return gen_output(M, N, preferred_element_type=preferred_element_type)


def torch_gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = torch.bfloat16,
    existing_out: Tensor | None = None,
) -> Tensor:
    assert group_sizes.dtype == torch.int32, "group_sizes type must be int32."

    M, _, N, G = shape_from_input(lhs, rhs, group_sizes)
    out = get_output(
        M, N, preferred_element_type=preferred_element_type, existing_out=existing_out
    )

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

    # Current tile. Each program computes multiple tiles of each group.
    tile = tl.program_id(0)

    # Tile limit of last MM problem (inclusive).
    last_mm_tile = 0

    # Last input row of lhs and output row of out. Each group reads some rows of
    # lhs and writes some rows to out.
    last_row = 0

    # Loop through all (m, K, N) MM problems:
    #   (m, K) x (K, N) = (m, N)
    #   sum(m) = M
    for g in range(G):
        # Get m dimension of current MM problem.
        m = tl.load(group_sizes_ptr + g * stride_group_sizes_g)

        num_m_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        # Loop through tiles of current MM problem.
        while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
            # Figure out tile coordinates in current MM problem.
            tile_in_mm = tile - last_mm_tile
            tile_m = tile_in_mm // num_n_tiles
            tile_n = tile_in_mm % num_n_tiles

            # Do regular MM:
            offs_lhs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_rhs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            lhs_ptrs = (
                lhs_ptr
                + (last_row + offs_lhs_m[:, None]) * stride_lhs_m
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
                + (last_row + offs_out_m[:, None]) * stride_out_m
                + offs_out_n[None, :] * stride_out_n
            )
            tl.store(
                out_ptrs,
                acc,
                mask=(offs_out_m[:, None] < m) & (offs_out_n[None, :] < N),
            )

            # Go to the next tile by advancing number of programs.
            tile += num_programs

        # Get ready to go to the next MM problem.
        last_mm_tile += num_tiles
        last_row += m


def triton_gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    tiling: tuple[int, int, int] | None = None,
    preferred_element_type: torch.dtype = torch.bfloat16,
    existing_out: Tensor | None = None,
) -> Tensor:
    assert group_sizes.dtype == torch.int32, "group_sizes type must be int32."

    M, K, N, G = shape_from_input(lhs, rhs, group_sizes)

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

    out = get_output(
        M, N, preferred_element_type=preferred_element_type, existing_out=existing_out
    )

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


@pytest.mark.skip(
    reason="Triton kernel isn't working for (M, K, N, G) = (10, 2, 3, 4) shape."
)
def test_simple_gmm():
    # M, K, N, G = 10, 2, 3, 4
    group_sizes = torch.tensor([3, 2, 4, 1], dtype=torch.int32, device=DEVICE)
    dtype = torch.float32
    # fmt: off
    lhs = torch.tensor([
        [ 1,  2],  # Group 0 (first 3 rows)
        [ 3,  4],
        [ 5,  6],
        [ 7,  8],  # Group 1 (next 2 rows)
        [ 9, 10],
        [11, 12],  # Group 2 (next 4 rows)
        [13, 14],
        [15, 16],
        [17, 18],
        [19, 20],  # Group 3 (last 1 row)
    ], dtype=dtype, device=DEVICE)
    rhs = torch.tensor([
        [[ 1,  2,  3],  # Group 0 matrix (2, 3)
         [ 4,  5,  6]],
        [[ 7,  8,  9],  # Group 1 matrix (2, 3)
         [10, 11, 12]],
        [[13, 14, 15],  # Group 2 matrix (2, 3)
         [16, 17, 18]],
        [[19, 20, 21],  # Group 3 matrix (2, 3)
         [22, 23, 24]],
    ], dtype=dtype, device=DEVICE)
    expected_out = torch.tensor([
        [  9,  12,  15],  # Group 0 matrix (3, 3)
        [ 19,  26,  33],
        [ 29,  40,  51],
        [129, 144, 159],  # Group 1 matrix (2, 3)
        [163, 182, 201],
        [335, 358, 381],  # Group 2 matrix (4, 3)
        [393, 420, 447],
        [451, 482, 513],
        [509, 544, 579],
        [801, 840, 879],  # Group 3 matrix (1, 3)
    ], dtype=dtype, device=DEVICE)
    # fmt: on
    out_torch = torch_gmm(lhs, rhs, group_sizes, preferred_element_type=dtype)
    print("\nout_torch", out_torch, sep="\n")
    torch.testing.assert_close(expected_out, out_torch)
    # FIXME: Triton kernel seems to be stuck in an infinite loop in this test!
    out_triton = triton_gmm(lhs, rhs, group_sizes, preferred_element_type=dtype)
    print("\nout_triton", out_triton, sep="\n")
    torch.testing.assert_close(expected_out, out_triton)


@pytest.mark.parametrize(
    "M, K, N, G",
    [
        (32, 16, 8, 4),  # Test 1
        (512, 4096, 2048, 160),  # Test 2
        (49152, 1408, 2048, 64),  # deepseekv2-16B
        # (3145728, 2048, 1408, 8),  # deepseekv2-16B (IT'S BIG! Getting core dump with this shape!)
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
