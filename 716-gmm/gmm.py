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

# Python standard library
import argparse
import random
import typing

# PyTorch
import torch
from torch import Tensor

# Triton
import triton
import triton.language as tl

# pytest
import pytest


# Global defaults.
# ------------------------------------------------------------------------------


DEVICE: torch.device | str = "cuda"


def dtype_from_str(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.strip().lower()
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        dtype_str[1:] if dtype_str[0] in {"i", "o"} else dtype_str
    ]


DTYPE_STR: str = "bf16"


DTYPE: torch.dtype = dtype_from_str(DTYPE_STR)


# TODO: Figure out a sensible tiling default.
TILING: tuple[int, int, int] = (64, 64, 64)


# Tensor creation functions.
# ------------------------------------------------------------------------------


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

    assert (
        len(group_sizes) == G
    ), f"Group sizes don't have {G} elements (it's {len(group_sizes)})."
    assert all(
        group_size > 0 for group_size in group_sizes
    ), "All group sizes must be positive."
    assert (
        sum(group_sizes) == M
    ), f"Group sizes don't add up to {M} (it's {sum(group_sizes)})."

    return group_sizes


def gen_input(
    M: int,
    K: int,
    N: int,
    G: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
    rng_seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    lhs = torch.randn((M, K), dtype=torch.float32, device=device).to(
        preferred_element_type
    )
    rhs = torch.randn((G, K, N), dtype=torch.float32, device=device).to(
        preferred_element_type
    )
    group_sizes = torch.tensor(
        random_group_sizes(M, G, rng_seed=rng_seed), dtype=torch.int32, device=device
    )

    return lhs, rhs, group_sizes


def gen_output(
    M: int,
    N: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
) -> Tensor:
    assert M > 0, f"Number of out rows M must be positive (M = {M})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."

    return torch.empty((M, N), dtype=preferred_element_type, device=device)


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


def shape_from_input(
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


def check_tiling(tiling: tuple[int, int, int]) -> tuple[int, int, int]:
    assert len(tiling) == 3, f"tiling must have 3 dimensions (it's = {len(tiling)})."
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
    return tiling


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


# PyTorch reference GMM implementation.
# ------------------------------------------------------------------------------


def torch_gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)
    M, _, N, G = shape_from_input(lhs, rhs, group_sizes)
    out = get_output(
        M,
        N,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        existing_out=existing_out,
    )

    offsets = torch.zeros(G + 1, dtype=torch.int32, device=lhs.device)
    torch.cumsum(group_sizes, dim=0, out=offsets[1:])

    for g in range(G):
        start_idx = offsets[g]
        end_idx = offsets[g + 1]
        out[start_idx:end_idx, :] = lhs[start_idx:end_idx, :] @ rhs[g]

    return out


# Kernel grid calculation.
# ------------------------------------------------------------------------------


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
    assert torch.all(group_sizes > 0).item(), "All group_sizes must be positive."
    num_m_tiles = (group_sizes + block_size_m - 1) // block_size_m
    assert torch.all(num_m_tiles > 0).item(), "All num_m_tiles must be positive."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = torch.sum(num_m_tiles * num_n_tiles).item()
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = int(min(num_sms(device=group_sizes.device), num_tiles))
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)


# Triton GMM simulation.
# TODO: Finish simulation logic.
# ------------------------------------------------------------------------------


def simulate_triton_gmm_kernel(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    out: Tensor,
    tiling: tuple[int, int, int] = TILING,
) -> None:
    def check_range(desc: str, x: int | Tensor, bits: int = 32):
        assert is_power_of_2(bits), f"Bit width must be a power of 2, it's {bits}."
        limit = 2 ** (bits - 1) - 1
        if isinstance(x, int):
            assert x >= 0, f"{desc} is < 0"
            assert x < limit, f"{desc} is >= max {bits}-bit value"
        else:
            assert isinstance(x, Tensor)
            assert torch.all(x >= 0).item(), f"{desc} is < 0"
            assert torch.all(x < limit).item(), f"{desc} is >= max {bits}-bit value"

    check_input_device_dtype(lhs, rhs, group_sizes)
    M, K, N, G = shape_from_input(lhs, rhs, group_sizes)
    stride_lhs_m, stride_lhs_k = lhs.stride()
    stride_rhs_g, stride_rhs_k, stride_rhs_n = rhs.stride()
    stride_out_m, stride_out_n = out.stride()
    block_size_m, block_size_k, block_size_n = check_tiling(tiling)
    num_programs = compute_grid(N, block_size_m, block_size_n, group_sizes)[0]

    for program_id in range(num_programs):
        tile = program_id
        check_range("tile (at initialization)", tile)

        # Tile limit of last MM problem (inclusive).
        last_mm_tile = 0
        check_range("last_mm_tile (at initialization)", last_mm_tile)

        # Last input row of lhs and output row of out. Each group reads some rows of
        # lhs and writes some rows to out.
        last_row = 0
        check_range("last_row (at initialization)", last_row)

        # Loop through all (m, K, N) MM problems:
        #   (m, K) x (K, N) = (m, N)
        #   sum(m) = M
        for g in range(G):
            # Get m dimension of current MM problem.
            m = int(group_sizes[g].item())
            check_range("m", m)

            num_m_tiles = triton.cdiv(m, block_size_m)
            check_range("num_m_tiles", num_m_tiles)
            num_n_tiles = triton.cdiv(N, block_size_n)
            check_range("num_n_tiles", num_n_tiles)
            num_tiles = num_m_tiles * num_n_tiles
            check_range("num_tiles", num_tiles)

            # Loop through tiles of current MM problem.
            while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
                # Figure out tile coordinates in current MM problem.
                tile_in_mm = tile - last_mm_tile
                check_range("tile_in_mm", tile_in_mm)
                tile_m = tile_in_mm // num_n_tiles
                assert tile_m >= 0, "tile_m < 0"
                check_range("tile_m", tile_m)
                tile_n = tile_in_mm % num_n_tiles
                check_range("tile_n", tile_n)

                # Do regular MM:
                # TODO: Implement simulation.

                # Go to the next tile by advancing number of programs.
                tile += num_programs
                check_range("tile (at update)", tile)

            # Get ready to go to the next MM problem.
            last_mm_tile += num_tiles
            check_range("last_mm_tile (at update)", last_mm_tile)
            last_row += m
            check_range("last_row (at update)", last_row)
            assert last_row > 0, "last_row <= 0 (at update)"
            assert last_row <= M, "last_row > M (at update)"

        assert last_row == M, "last_row != M (at end)"


# Triton GMM implementation.
# ------------------------------------------------------------------------------


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
    tl.assume(stride_out_m > 0)
    tl.assume(stride_out_n > 0)

    stride_lhs_m = stride_lhs_m.to(tl.int64)
    stride_lhs_k = stride_lhs_k.to(tl.int64)
    stride_rhs_g = stride_rhs_g.to(tl.int64)
    stride_rhs_k = stride_rhs_k.to(tl.int64)
    stride_rhs_n = stride_rhs_n.to(tl.int64)
    stride_out_m = stride_out_m.to(tl.int64)
    stride_out_n = stride_out_n.to(tl.int64)

    # Current tile. Each program computes multiple tiles of each group.
    tile = tl.program_id(0)
    tl.device_assert(tile >= 0, "tile < 0 (at initialization)")

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
        m = tl.load(group_sizes_ptr + g)
        tl.device_assert(m > 0, "m <= 0")

        num_m_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        tl.device_assert(num_m_tiles > 0, "num_m_tiles <= 0")
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        tl.device_assert(num_n_tiles > 0, "num_n_tiles <= 0")
        num_tiles = num_m_tiles * num_n_tiles
        tl.device_assert(num_tiles > 0, "num_tiles <= 0")

        # Loop through tiles of current MM problem.
        while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
            # Figure out tile coordinates in current MM problem.
            tile_in_mm = tile - last_mm_tile
            tl.device_assert(tile_in_mm >= 0, "tile_in_mm < 0")
            tile_m = tile_in_mm // num_n_tiles
            tl.device_assert(tile_m >= 0, "tile_m < 0")
            tl.device_assert(tile_m < num_m_tiles, "tile_m >= num_m_tiles")
            tile_n = tile_in_mm % num_n_tiles
            tl.device_assert(tile_n >= 0, "tile_n < 0")
            tl.device_assert(tile_n < num_n_tiles, "tile_n >= num_n_tiles")

            # Do regular MM:

            tl.device_assert(tile_m * BLOCK_SIZE_M >= 0, "tile_m * BLOCK_SIZE_M < 0")
            offs_lhs_m = (
                tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            ) % m
            tl.device_assert(tl.min(offs_lhs_m >= 0) == 1, "offs_lhs_m < 0")

            tl.device_assert(tile_n * BLOCK_SIZE_N >= 0, "tile_n * BLOCK_SIZE_N < 0")
            offs_rhs_n = (
                tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            ) % N
            tl.device_assert(tl.min(offs_rhs_n >= 0) == 1, "offs_rhs_n < 0")

            offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

            lhs_offs_0 = last_row + offs_lhs_m[:, None]
            tl.device_assert(tl.min(lhs_offs_0 >= 0) == 1, "lhs_offs_0 < 0")
            tl.device_assert(
                lhs_offs_0.dtype == last_row.dtype, "wrong lhs_offs_0 type"
            )
            lhs_offs_1 = lhs_offs_0 * stride_lhs_m
            tl.device_assert(tl.min(lhs_offs_1 >= 0) == 1, "lhs_offs_1 < 0")
            tl.device_assert(
                lhs_offs_1.dtype == stride_lhs_m.dtype, "wrong lhs_offs_1 type"
            )
            lhs_offs_2 = offs_k[None, :] * stride_lhs_k
            tl.device_assert(tl.min(lhs_offs_2 >= 0) == 1, "lhs_offs_2 < 0")
            tl.device_assert(
                lhs_offs_2.dtype == stride_lhs_k.dtype, "wrong lhs_offs_2 type"
            )
            lhs_offs_3 = lhs_offs_1 + lhs_offs_2
            tl.device_assert(tl.min(lhs_offs_3 >= 0) == 1, "lhs_offs_3 < 0")
            tl.device_assert(
                lhs_offs_3.dtype == stride_lhs_m.dtype, "wrong lhs_offs_3 type"
            )
            lhs_ptrs = lhs_ptr + lhs_offs_3

            rhs_offs_1 = g * stride_rhs_g
            tl.device_assert(rhs_offs_1 >= 0, "rhs_offs_1 < 0")
            tl.device_assert(
                rhs_offs_1.dtype == stride_rhs_g.dtype, "wrong rhs_offs_1 type"
            )
            rhs_offs_2 = offs_k[:, None] * stride_rhs_k
            tl.device_assert(tl.min(rhs_offs_2 >= 0) == 1, "rhs_offs_2 < 0")
            tl.device_assert(
                rhs_offs_2.dtype == stride_rhs_k.dtype, "wrong rhs_offs_2 type"
            )
            rhs_offs_3 = offs_rhs_n[None, :] * stride_rhs_n
            tl.device_assert(tl.min(rhs_offs_3 >= 0) == 1, "rhs_offs_3 < 0")
            tl.device_assert(
                rhs_offs_3.dtype == stride_rhs_n.dtype, "wrong rhs_offs_3 type"
            )
            rhs_offs_4 = rhs_offs_1 + rhs_offs_2 + rhs_offs_3
            tl.device_assert(tl.min(rhs_offs_4 >= 0) == 1, "rhs_offs_4 < 0")
            tl.device_assert(
                rhs_offs_4.dtype == stride_rhs_g.dtype, "wrong rhs_offs_4 type"
            )
            rhs_ptrs = rhs_ptr + rhs_offs_4

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                k_mask_limit = K - k.to(tl.int64) * BLOCK_SIZE_K
                tl.device_assert(k_mask_limit > 0, "k_mask_limit <= 0")
                lhs = tl.load(lhs_ptrs, mask=offs_k[None, :] < k_mask_limit, other=0)
                rhs = tl.load(rhs_ptrs, mask=offs_k[:, None] < k_mask_limit, other=0)

                acc += tl.dot(lhs, rhs, input_precision="ieee")

                lhs_step = BLOCK_SIZE_K * stride_lhs_k
                tl.device_assert(lhs_step > 0, "lhs_step <= 0")
                tl.device_assert(
                    lhs_step.dtype == stride_lhs_k.dtype, "wrong lhs_step type"
                )
                lhs_ptrs += lhs_step

                rhs_step = BLOCK_SIZE_K * stride_rhs_k
                tl.device_assert(rhs_step > 0, "rhs_step <= 0")
                tl.device_assert(
                    rhs_step.dtype == stride_rhs_k.dtype, "wrong rhs_step type"
                )
                rhs_ptrs += rhs_step

            acc = acc.to(out_ptr.type.element_ty)

            # tile_m * BLOCK_SIZE_M >= 0 was already checked
            offs_out_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            tl.device_assert(tl.min(offs_out_m >= 0) == 1, "offs_out_m < 0")

            # tile_n * BLOCK_SIZE_N >= 0 was already checked
            offs_out_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            tl.device_assert(tl.min(offs_out_n >= 0) == 1, "offs_out_n < 0")

            out_offs_0 = last_row + offs_out_m[:, None]
            tl.device_assert(tl.min(out_offs_0 >= 0) == 1, "out_offs_0 < 0")
            tl.device_assert(
                out_offs_0.dtype == last_row.dtype, "wrong out_offs_0 type"
            )
            out_offs_1 = out_offs_0 * stride_out_m
            tl.device_assert(tl.min(out_offs_1 >= 0) == 1, "out_offs_1 < 0")
            tl.device_assert(
                out_offs_1.dtype == stride_out_m.dtype, "wrong out_offs_1 type"
            )
            out_offs_2 = offs_out_n[None, :] * stride_out_n
            tl.device_assert(tl.min(out_offs_2 >= 0) == 1, "out_offs_2 < 0")
            tl.device_assert(
                out_offs_2.dtype == stride_out_n.dtype, "wrong out_offs_2 type"
            )
            out_offs_3 = out_offs_1 + out_offs_2
            tl.device_assert(tl.min(out_offs_3 >= 0) == 1, "out_offs_3 < 0")
            tl.device_assert(
                out_offs_3.dtype == stride_out_m.dtype, "wrong out_offs_3 type"
            )
            out_ptrs = out_ptr + out_offs_3
            tl.store(
                out_ptrs,
                acc,
                mask=(offs_out_m[:, None] < m) & (offs_out_n[None, :] < N),
            )

            # Go to the next tile by advancing number of programs.
            tile += tl.num_programs(0)
            tl.device_assert(tile > 0, "tile <= 0 (at update)")

        # Get ready to go to the next MM problem.
        last_mm_tile += num_tiles
        tl.device_assert(last_mm_tile > 0, "last_mm_tile <= 0 (at update)")
        last_row += m
        tl.device_assert(last_row > 0, "last_row <= 0 (at update)")
        tl.device_assert(last_row <= M, "last_row > M (at update)")

    tl.device_assert(last_row == M, "last_row != M (at end)")


def triton_gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    tiling: tuple[int, int, int] = TILING,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)
    M, K, N, G = shape_from_input(lhs, rhs, group_sizes)
    block_size_m, block_size_k, block_size_n = check_tiling(tiling)
    out = get_output(
        M,
        N,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        existing_out=existing_out,
    )

    grid = compute_grid(N, block_size_m, block_size_n, group_sizes)
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
        *out.stride(),
        # Meta-parameters:
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_N=block_size_n,
    )

    return out


# Unit tests.
# ------------------------------------------------------------------------------


@pytest.mark.skip(reason="Triton kernel isn't working with fp32 input type.")
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


# fmt: off
@pytest.mark.parametrize(
    "M, K, N, G",
    [
        (     10,     2,     3,   4),  # same shape of test_simple_gmm
        (     32,    16,     8,   4),  # Test 1
        (    512,  4096,  2048, 160),  # Test 2
        (  49152,  1408,  2048,  64),  # deepseekv2-16B

        (3145728,  2048,  1408,   8),  # deepseekv2-16B (IT'S BIG! I was getting core dump with this shape! lhs => 12 GB, out => 8.25 GB)
      # (1867775,  2048,  1408,   8),  #   lhs => 7.12 GB, out => 4.90 GB
      # (1857944,  2048,  1408,   8),  #   lhs => 7.09 GB, out => 4.87 GB
      # (1853029,  2048,  1408,   8),  #   lhs => 7.07 GB, out => 4.86 GB
      # (1850571,  2048,  1408,   8),  #   lhs => 7.06 GB, out => 4.85 GB
      # (1849342,  2048,  1408,   8),  #   lhs => 7.05 GB, out => 4.85 GB
      # (1848114,  2048,  1408,   8),  #   lhs => 7.05 GB, out => 4.85 GB
      # (1808793,  2048,  1408,   8),  #   lhs => 6.90 GB, out => 4.74 GB
      # (1730150,  2048,  1408,   8),  #   lhs => 6.60 GB, out => 4.54 GB
      # (1525202,  2048,  1408,   8),  #   lhs => 5.82 GB, out => 4.000001 GB
      # (1525201,  2048,  1408,   8),  #   lhs => 5.82 GB, out => 3.999999 GB
      # (1048577,  2048,  1408,   8),  #   lhs => 4 GB + 4 KB, out => 2.75 GB
      # (1048576,  2048,  1408,   8),  #   lhs => 4 GB, out => 2.75 GB

        ( 393216,  2048,  1408,  64),  # deepseekv2-16B
        (  32768,  6144, 16384,   8),  # Mixtral 8x22B proxy model
        (  32768, 16384,  6144,   8),  # Mixtral 8x22B proxy model
    ],
)
# fmt: on
@pytest.mark.parametrize("in_dtype_str", ["ifp16", "ibf16", "ifp32"])
@pytest.mark.parametrize("out_dtype_str", ["ofp16", "obf16", "ofp32"])
@pytest.mark.parametrize("rng_seed", [0, 77, 121])
def test_gmm(M: int, K: int, N: int, G: int, in_dtype_str: str, out_dtype_str: str, rng_seed: int):
    in_dtype = dtype_from_str(in_dtype_str)
    if in_dtype == torch.float32:
        pytest.skip("Triton kernel isn't working with fp32 input type.")
    out_dtype = dtype_from_str(out_dtype_str)
    lhs, rhs, group_sizes = gen_input(
        M, K, N, G, preferred_element_type=in_dtype, rng_seed=rng_seed
    )
    out_torch = torch_gmm(lhs, rhs, group_sizes, preferred_element_type=out_dtype)
    out_triton = gen_output(M, N, preferred_element_type=out_dtype)
    simulate_triton_gmm_kernel(lhs, rhs, group_sizes, out_triton)
    out_triton = triton_gmm(lhs, rhs, group_sizes, preferred_element_type=out_dtype, existing_out=out_triton)
    torch.testing.assert_close(out_torch, out_triton, atol=5e-3, rtol=1e-2)


# Command line interface parsing.
# TODO: Add types to command line interface.
# ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    def positive_int(value: str) -> int:
        try:
            int_value = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
        if int_value <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
        return int_value

    parser = argparse.ArgumentParser(description="run GMM Triton kernel")
    parser.add_argument("M", type=positive_int, help="number of rows")
    parser.add_argument("K", type=positive_int, help="shared dimension")
    parser.add_argument("N", type=positive_int, help="number of columns")
    parser.add_argument("G", type=positive_int, help="number of groups")
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="random seed for input generation (default: 0)",
    )
    return parser.parse_args()


# Main function: entry point.
# TODO: Implement benchmark.
# ------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    lhs, rhs, group_sizes = gen_input(
        args.M, args.K, args.N, args.G, rng_seed=args.rng_seed
    )
    triton_gmm(lhs, rhs, group_sizes)


if __name__ == "__main__":
    main()
