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


# Default device.
DEVICE: torch.device | str = "cuda"


# Supported data types, as strings.
SUPPORTED_DTYPES_STR: set[str] = {"fp16", "bf16", "fp32"}


# Convert string data type to PyTorch data type.
def dtype_from_str(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.strip().lower()
    dtype_str = dtype_str[1:] if dtype_str[0] in {"i", "o"} else dtype_str
    assert (
        dtype_str in SUPPORTED_DTYPES_STR
    ), "String data type isn't in set of supported string data types."
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        dtype_str
    ]


# Supported data types, as PyTorch types.
SUPPORTED_DTYPES: set[torch.dtype] = {
    dtype_from_str(dtype_str) for dtype_str in SUPPORTED_DTYPES_STR
}


def str_from_dtype(dtype: torch.dtype) -> str:
    assert (
        dtype in SUPPORTED_DTYPES
    ), "PyTorch data type isn't in set of supported PyTorch data types."
    return {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]


# Default data type, as string.
DTYPE_STR: str = "bf16"
assert (
    DTYPE_STR in SUPPORTED_DTYPES_STR
), "Default string data type isn't in set of supported string data types."
# Default data type, as PyTorch type.
DTYPE: torch.dtype = dtype_from_str(DTYPE_STR)


# Default RNG seed for input generation.
RNG_SEED: int = 0


# TODO: Figure out a sensible tiling default.
TILING: tuple[int, int, int] = (64, 64, 64)


# Tensor creation functions.
# ------------------------------------------------------------------------------


def random_group_sizes(
    M: int, G: int, device: torch.device | str = DEVICE, rng_seed: int | None = RNG_SEED
) -> Tensor:
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

    return torch.tensor(group_sizes, dtype=torch.int32, device=device)


def gen_input(
    M: int,
    K: int,
    N: int,
    G: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
    rng_seed: int | None = RNG_SEED,
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
    group_sizes = random_group_sizes(M, G, device=device, rng_seed=rng_seed)

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

    stride_lhs_type = tl.int64
    stride_rhs_type = tl.int64
    stride_out_type = tl.int64

    stride_lhs_m = stride_lhs_m.to(stride_lhs_type)
    stride_lhs_k = stride_lhs_k.to(stride_lhs_type)
    stride_rhs_g = stride_rhs_g.to(stride_rhs_type)
    stride_rhs_k = stride_rhs_k.to(stride_rhs_type)
    stride_rhs_n = stride_rhs_n.to(stride_rhs_type)
    stride_out_m = stride_out_m.to(stride_out_type)
    stride_out_n = stride_out_n.to(stride_out_type)

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
            tl.device_assert(tile_n * BLOCK_SIZE_N >= 0, "tile_n * BLOCK_SIZE_N < 0")

            offs_lhs_m = (
                tile_m.to(stride_lhs_type) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            ) % m
            tl.device_assert(
                offs_lhs_m.dtype == stride_lhs_type, "wrong offs_lhs_m type"
            )
            offs_rhs_n = (
                tile_n.to(stride_rhs_type) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            ) % N
            tl.device_assert(
                offs_rhs_n.dtype == stride_lhs_type, "wrong offs_rhs_n type"
            )
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            lhs_offs_0 = last_row + offs_lhs_m[:, None]
            tl.device_assert(
                lhs_offs_0.dtype == stride_lhs_type, "wrong lhs_offs_0 type"
            )
            lhs_offs_1 = lhs_offs_0 * stride_lhs_m
            tl.device_assert(
                lhs_offs_1.dtype == stride_lhs_type, "wrong lhs_offs_1 type"
            )
            lhs_offs_2 = offs_k[None, :] * stride_lhs_k
            tl.device_assert(
                lhs_offs_2.dtype == stride_lhs_type, "wrong lhs_offs_2 type"
            )
            lhs_offs_3 = lhs_offs_1 + lhs_offs_2
            tl.device_assert(
                lhs_offs_3.dtype == stride_lhs_type, "wrong lhs_offs_3 type"
            )
            lhs_ptrs = lhs_ptr + lhs_offs_3

            rhs_offs_1 = g * stride_rhs_g
            tl.device_assert(
                rhs_offs_1.dtype == stride_rhs_type, "wrong rhs_offs_1 type"
            )
            rhs_offs_2 = offs_k[:, None] * stride_rhs_k
            tl.device_assert(
                rhs_offs_2.dtype == stride_rhs_type, "wrong rhs_offs_2 type"
            )
            rhs_offs_3 = offs_rhs_n[None, :] * stride_rhs_n
            tl.device_assert(
                rhs_offs_3.dtype == stride_rhs_type, "wrong rhs_offs_3 type"
            )
            rhs_offs_4 = rhs_offs_1 + rhs_offs_2 + rhs_offs_3
            tl.device_assert(
                rhs_offs_4.dtype == stride_rhs_type, "wrong rhs_offs_4 type"
            )
            rhs_ptrs = rhs_ptr + rhs_offs_4

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                k_mask_limit = K - k * BLOCK_SIZE_K
                lhs = tl.load(lhs_ptrs, mask=offs_k[None, :] < k_mask_limit, other=0)
                rhs = tl.load(rhs_ptrs, mask=offs_k[:, None] < k_mask_limit, other=0)

                acc += tl.dot(lhs, rhs, input_precision="ieee")

                lhs_step = BLOCK_SIZE_K * stride_lhs_k
                tl.device_assert(lhs_step > 0, "lhs_step <= 0")
                tl.device_assert(
                    lhs_step.dtype == stride_lhs_type, "wrong lhs_step type"
                )
                lhs_ptrs += lhs_step

                rhs_step = BLOCK_SIZE_K * stride_rhs_k
                tl.device_assert(rhs_step > 0, "rhs_step <= 0")
                tl.device_assert(
                    rhs_step.dtype == stride_rhs_type, "wrong rhs_step type"
                )
                rhs_ptrs += rhs_step

            acc = acc.to(out_ptr.type.element_ty)

            offs_out_m = tile_m.to(stride_out_type) * BLOCK_SIZE_M + tl.arange(
                0, BLOCK_SIZE_M
            )
            tl.device_assert(
                offs_out_m.dtype == stride_out_type, "wrong offs_out_m type"
            )
            offs_out_n = tile_n.to(stride_out_type) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            )
            tl.device_assert(
                offs_out_n.dtype == stride_out_type, "wrong offs_out_n type"
            )

            out_offs_0 = last_row + offs_out_m[:, None]
            tl.device_assert(
                out_offs_0.dtype == stride_out_type, "wrong out_offs_0 type"
            )
            out_offs_1 = out_offs_0 * stride_out_m
            tl.device_assert(
                out_offs_1.dtype == stride_out_type, "wrong out_offs_1 type"
            )
            out_offs_2 = offs_out_n[None, :] * stride_out_n
            tl.device_assert(
                out_offs_2.dtype == stride_out_type, "wrong out_offs_2 type"
            )
            out_offs_3 = out_offs_1 + out_offs_2
            tl.device_assert(
                out_offs_3.dtype == stride_out_type, "wrong out_offs_3 type"
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


# GMM shapes used only for test purposes,
# fmt: off
TEST_ONLY_SHAPES: list[tuple[int, int, int, int]] = [
    ( 10,    2,    3,   4),  # same shape of test_simple_gmm
    ( 32,   16,    8,   4),  # Test 1
    (512, 4096, 2048, 160),  # Test 2
]
# fmt: on


# Real GMM shapes, used by real models.
# fmt: off
REAL_SHAPES: list[tuple[int, int, int, int]] = [
    (  49152,  1408,  2048,  64),  # deepseekv2-16B
    (3145728,  2048,  1408,   8),  # deepseekv2-16B (IT'S BIG! I was getting core dump with this shape! lhs => 12 GB, out => 8.25 GB)
    ( 393216,  2048,  1408,  64),  # deepseekv2-16B
    (  32768,  6144, 16384,   8),  # Mixtral 8x22B proxy model
    (  32768, 16384,  6144,   8),  # Mixtral 8x22B proxy model
]
# fmt: on


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


@pytest.mark.parametrize("M, K, N, G", TEST_ONLY_SHAPES + REAL_SHAPES)
@pytest.mark.parametrize(
    "in_dtype_str", {"i" + dtype_str for dtype_str in SUPPORTED_DTYPES_STR}
)
@pytest.mark.parametrize(
    "out_dtype_str", {"o" + dtype_str for dtype_str in SUPPORTED_DTYPES_STR}
)
@pytest.mark.parametrize("rng_seed", [0, 77, 121])
def test_gmm(
    M: int, K: int, N: int, G: int, in_dtype_str: str, out_dtype_str: str, rng_seed: int
):
    in_dtype = dtype_from_str(in_dtype_str)
    if in_dtype == torch.float32:
        pytest.skip("Triton kernel isn't working with fp32 input type.")
    out_dtype = dtype_from_str(out_dtype_str)
    lhs, rhs, group_sizes = gen_input(
        M, K, N, G, preferred_element_type=in_dtype, rng_seed=rng_seed
    )
    out_torch = torch_gmm(lhs, rhs, group_sizes, preferred_element_type=out_dtype)
    out_triton = triton_gmm(lhs, rhs, group_sizes, preferred_element_type=out_dtype)
    torch.testing.assert_close(out_torch, out_triton, atol=5e-3, rtol=1e-2)


# Benchmark.
# ------------------------------------------------------------------------------


def benchmark_triton_gmm(
    target_shape: tuple[int, int, int, int] | None = None,
    in_dtype: torch.dtype = DTYPE,
    out_dtype: torch.dtype = DTYPE,
    rng_seed: int = RNG_SEED,
) -> None:
    in_dtype_str = str_from_dtype(in_dtype)
    out_dtype_str = str_from_dtype(out_dtype)
    dtypes_desc = f"i{in_dtype_str}_o{out_dtype_str}"
    triton_provider = f"triton_{dtypes_desc}"

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "K", "N", "G"],
            x_vals=[target_shape] if target_shape is not None else REAL_SHAPES,
            line_arg="provider",
            line_vals=[triton_provider],
            line_names=[triton_provider],
            plot_name=f"triton_gmm_perf_{dtypes_desc}",
            args={},
            ylabel="TFLOPS",
        )
    )
    def benchmark(M: int, K: int, N: int, G: int, provider: str):
        assert "triton" in provider, f"GMM provider isn't triton, it's {provider}."

        print(f"\t\t(M, K, N, G) = {(M, K, N, G)}")

        lhs, rhs, group_sizes_0 = gen_input(
            M, K, N, G, preferred_element_type=in_dtype, rng_seed=rng_seed
        )
        out = gen_output(M, N, preferred_element_type=out_dtype)

        distinct_group_sizes = 10
        group_sizes = [group_sizes_0] + [
            random_group_sizes(M, G, rng_seed=None)
            for _ in range(distinct_group_sizes - 1)
        ]
        assert (
            len(group_sizes) == distinct_group_sizes
        ), f"Expecting {distinct_group_sizes} distinct group sizes."

        quantiles = [0.5, 0.2, 0.8]
        p50_s_sum = 0.0
        p20_s_sum = 0.0
        p80_s_sum = 0.0
        tops_sum = 0.0

        for group_sizes_g in group_sizes:
            print(f"\t\t\tgroup_sizes = {group_sizes_g[:8].tolist()}")

            p50_ms, p20_ms, p80_ms = triton.testing.do_bench(
                lambda: triton_gmm(
                    lhs,
                    rhs,
                    group_sizes_g,
                    preferred_element_type=out_dtype,
                    existing_out=out,
                ),
                quantiles=quantiles,
            )
            p50_s_sum += p50_ms * 1e-3
            p20_s_sum += p20_ms * 1e-3
            p80_s_sum += p80_ms * 1e-3
            tops_sum += torch.sum(1e-12 * group_sizes_g * N * (K + (K - 1))).item()

        p50_tflops = tops_sum / p50_s_sum
        p20_tflops = tops_sum / p20_s_sum
        p80_tflops = tops_sum / p80_s_sum
        return p50_tflops, p80_tflops, p20_tflops

    print(
        f"Benchmarking...\n\tinput_type = {in_dtype_str}, output_type = {out_dtype_str}"
    )
    benchmark.run(show_plots=False, print_data=True)


# Standalone kernel runner.
# It's useful for `rocprof` profiling and collecting ATT traces.
# ------------------------------------------------------------------------------


def run_triton_gmm(
    M: int,
    K: int,
    N: int,
    G: int,
    in_dtype: torch.dtype = DTYPE,
    out_dtype: torch.dtype = DTYPE,
    rng_seed: int = RNG_SEED,
) -> None:
    lhs, rhs, group_sizes = gen_input(
        M, K, N, G, preferred_element_type=in_dtype, rng_seed=rng_seed
    )
    triton_gmm(lhs, rhs, group_sizes, preferred_element_type=out_dtype)


# Command line interface parsing.
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
        "--input-type",
        choices=SUPPORTED_DTYPES_STR,
        default=DTYPE_STR,
        help=f"input data type (default: {DTYPE_STR})",
    )
    parser.add_argument(
        "--output-type",
        choices=SUPPORTED_DTYPES_STR,
        default=DTYPE_STR,
        help=f"output data type (default: {DTYPE_STR})",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=RNG_SEED,
        help=f"random seed for input generation (default: {RNG_SEED})",
    )
    return parser.parse_args()


# Main function: entry point.
# TODO: Implement benchmark.
# ------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    in_dtype = dtype_from_str(args.input_type)
    out_dtype = dtype_from_str(args.output_type)

    run_triton_gmm(
        args.M,
        args.K,
        args.N,
        args.G,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        rng_seed=args.rng_seed,
    )

    # benchmark_triton_gmm(in_dtype=in_dtype, out_dtype=out_dtype, rng_seed=args.rng_seed)


if __name__ == "__main__":
    main()
