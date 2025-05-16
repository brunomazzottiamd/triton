# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import typing

# Triton
import triton
import triton.language as tl


# Triton GMM kernel.
# ------------------------------------------------------------------------------


@triton.jit
@typing.no_type_check
def triton_gmm_kernel_core(
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

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    tl.device_assert(num_n_tiles > 0, "num_m_tiles <= 0")

    # stride_lhs_k = 1 when lhs is row-major
    lhs_step = BLOCK_SIZE_K * stride_lhs_k
    tl.device_assert(lhs_step > 0, "lhs_step <= 0")

    # stride_rhs_k = 1 when rhs is column-major
    rhs_step = BLOCK_SIZE_K * stride_rhs_k
    tl.device_assert(rhs_step > 0, "rhs_step <= 0")

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
        # m can be zero if group is empty
        tl.device_assert(m >= 0, "m < 0")

        num_m_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        # num_m_tiles can be zero if group is empty
        tl.device_assert(num_m_tiles >= 0, "num_m_tiles < 0")

        num_tiles = num_m_tiles * num_n_tiles
        # num_tiles can be zero if group is empty
        tl.device_assert(num_tiles >= 0, "num_tiles < 0")

        # Loop through tiles of current MM problem.
        while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
            # Figure out tile coordinates in current MM problem.
            tile_in_mm = tile - last_mm_tile
            tl.device_assert(tile_in_mm >= 0, "tile_in_mm < 0")

            if GROUP_SIZE_M == 1:
                tile_m = tile_in_mm // num_n_tiles
                tile_n = tile_in_mm % num_n_tiles
            else:
                # Re-order program ID for better L2 performance.
                num_tiles_in_group = GROUP_SIZE_M * num_n_tiles
                group_id = tile_in_mm // num_tiles_in_group
                first_tile_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_m_tiles - first_tile_m, GROUP_SIZE_M)
                tile_m = first_tile_m + (tile_in_mm % group_size_m)
                tile_n = (tile_in_mm % num_tiles_in_group) // group_size_m

            tl.device_assert(tile_m >= 0, "tile_m < 0")
            tl.device_assert(tile_m < num_m_tiles, "tile_m >= num_m_tiles")
            tl.device_assert(tile_n >= 0, "tile_n < 0")
            tl.device_assert(tile_n < num_n_tiles, "tile_n >= num_n_tiles")

            # Do regular MM:

            tl.device_assert(tile_m * BLOCK_SIZE_M >= 0, "tile_m * BLOCK_SIZE_M < 0")
            tl.device_assert(tile_n * BLOCK_SIZE_N >= 0, "tile_n * BLOCK_SIZE_N < 0")

            offs_lhs_m = (
                tile_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            ) % m
            tl.device_assert(offs_lhs_m.dtype == tl.int64, "wrong offs_lhs_m type")
            offs_rhs_n = (
                tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            ) % N
            tl.device_assert(offs_rhs_n.dtype == tl.int64, "wrong offs_rhs_n type")
            offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

            lhs_offs_0 = last_row + offs_lhs_m[:, None]
            tl.device_assert(lhs_offs_0.dtype == tl.int64, "wrong lhs_offs_0 type")
            # stride_lhs_m = 1 when lhs is column-major [A = tl.multiple_of(A, (1, 16))]
            lhs_offs_1 = lhs_offs_0 * stride_lhs_m
            tl.device_assert(lhs_offs_1.dtype == tl.int64, "wrong lhs_offs_1 type")
            # stride_lhs_k = 1 when lhs is row-major [A = tl.multiple_of(A, (16, 1))]
            lhs_offs_2 = offs_k[None, :] * stride_lhs_k
            tl.device_assert(lhs_offs_2.dtype == tl.int64, "wrong lhs_offs_2 type")
            lhs_offs_3 = lhs_offs_1 + lhs_offs_2
            tl.device_assert(lhs_offs_3.dtype == tl.int64, "wrong lhs_offs_3 type")
            lhs_ptrs = lhs_ptr + lhs_offs_3

            # stride_rhs_g is always K * N
            rhs_offs_1 = g.to(tl.int64) * stride_rhs_g
            tl.device_assert(rhs_offs_1.dtype == tl.int64, "wrong rhs_offs_1 type")
            # stride_rhs_k = 1 when rhs is column-major [???]
            rhs_offs_2 = offs_k[:, None] * stride_rhs_k
            tl.device_assert(rhs_offs_2.dtype == tl.int64, "wrong rhs_offs_2 type")
            # stride_rhs_n = 1 when rhs is row-major [???]
            rhs_offs_3 = offs_rhs_n[None, :] * stride_rhs_n
            tl.device_assert(rhs_offs_3.dtype == tl.int64, "wrong rhs_offs_3 type")
            rhs_offs_4 = rhs_offs_1 + rhs_offs_2 + rhs_offs_3
            tl.device_assert(rhs_offs_4.dtype == tl.int64, "wrong rhs_offs_4 type")
            rhs_ptrs = rhs_ptr + rhs_offs_4

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                if K_DIVISIBLE_BY_BLOCK_SIZE_K:
                    lhs = tl.load(lhs_ptrs)
                    rhs = tl.load(rhs_ptrs)
                else:
                    k_mask_limit = K - k * BLOCK_SIZE_K
                    lhs = tl.load(
                        lhs_ptrs, mask=offs_k[None, :] < k_mask_limit, other=0
                    )
                    rhs = tl.load(
                        rhs_ptrs, mask=offs_k[:, None] < k_mask_limit, other=0
                    )

                acc += tl.dot(lhs, rhs, input_precision="ieee")

                lhs_ptrs += lhs_step
                rhs_ptrs += rhs_step

            acc = acc.to(out_ptr.type.element_ty)

            offs_out_m = tile_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            tl.device_assert(offs_out_m.dtype == tl.int64, "wrong offs_out_m type")
            offs_out_n = tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            tl.device_assert(offs_out_n.dtype == tl.int64, "wrong offs_out_n type")

            out_offs_0 = last_row + offs_out_m[:, None]
            tl.device_assert(out_offs_0.dtype == tl.int64, "wrong out_offs_0 type")
            # stride_out_m = 1 when out is column-major
            out_offs_1 = out_offs_0 * stride_out_m
            tl.device_assert(out_offs_1.dtype == tl.int64, "wrong out_offs_1 type")
            # stride_out_n = 1 when out is row-major
            out_offs_2 = offs_out_n[None, :] * stride_out_n
            tl.device_assert(out_offs_2.dtype == tl.int64, "wrong out_offs_2 type")
            out_offs_3 = out_offs_1 + out_offs_2
            tl.device_assert(out_offs_3.dtype == tl.int64, "wrong out_offs_3 type")
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
        # last_mm_tile can be zero if group 0 is skipped
        tl.device_assert(last_mm_tile >= 0, "last_mm_tile < 0 (at update)")
        last_row += m
        # last_row can be zero if group 0 is skipped
        tl.device_assert(last_row >= 0, "last_row < 0 (at update)")
        tl.device_assert(last_row <= M, "last_row > M (at update)")

    tl.device_assert(last_row <= M, "last_row > M (at end)")
