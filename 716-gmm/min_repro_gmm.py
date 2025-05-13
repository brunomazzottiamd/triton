import torch

import triton
import triton.language as tl


@triton.jit
def triton_gmm_kernel_core(
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    # stride_rhs_n,  # <= kernel works if stride_rhs_n isn't tl.constexpr
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    stride_lhs_m: tl.constexpr,
    stride_lhs_k: tl.constexpr,
    stride_rhs_g: tl.constexpr,
    stride_rhs_k: tl.constexpr,
    stride_rhs_n: tl.constexpr,  # <= kernel doesn't work if stride_rhs_n is tl.constexpr
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr,
):
    tl.static_assert(M > 0)
    tl.static_assert(K > 0)
    tl.static_assert(N > 0)
    tl.static_assert(G > 0)

    tl.assume(M > 0)
    tl.assume(K > 0)
    tl.assume(N > 0)
    tl.assume(G > 0)

    tl.static_assert(stride_lhs_m > 0)
    tl.static_assert(stride_lhs_k > 0)
    tl.static_assert(stride_rhs_g > 0)
    tl.static_assert(stride_rhs_k > 0)
    # tl.static_assert(stride_rhs_n > 0)
    tl.static_assert(stride_out_m > 0)
    tl.static_assert(stride_out_n > 0)

    tl.assume(stride_lhs_m > 0)
    tl.assume(stride_lhs_k > 0)
    tl.assume(stride_rhs_g > 0)
    tl.assume(stride_rhs_k > 0)
    tl.assume(stride_rhs_n > 0)
    tl.assume(stride_out_m > 0)
    tl.assume(stride_out_n > 0)

    # tl.cdiv(N, BLOCK_SIZE_N) doesn't play well with tl.constexpr.
    num_n_tiles: tl.constexpr = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    tl.static_assert(num_n_tiles > 0)
    tl.assume(num_n_tiles > 0)

    lhs_step: tl.constexpr = BLOCK_SIZE_K * stride_lhs_k
    tl.static_assert(lhs_step > 0)
    tl.assume(lhs_step > 0)

    rhs_step: tl.constexpr = BLOCK_SIZE_K * stride_rhs_k
    tl.static_assert(rhs_step > 0)
    tl.assume(rhs_step > 0)

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
            lhs_offs_1 = lhs_offs_0 * stride_lhs_m
            tl.device_assert(lhs_offs_1.dtype == tl.int64, "wrong lhs_offs_1 type")
            lhs_offs_2 = offs_k[None, :] * stride_lhs_k
            tl.device_assert(lhs_offs_2.dtype == tl.int64, "wrong lhs_offs_2 type")
            lhs_offs_3 = lhs_offs_1 + lhs_offs_2
            tl.device_assert(lhs_offs_3.dtype == tl.int64, "wrong lhs_offs_3 type")
            lhs_ptrs = lhs_ptr + lhs_offs_3

            rhs_offs_1 = g.to(tl.int64) * stride_rhs_g
            tl.device_assert(rhs_offs_1.dtype == tl.int64, "wrong rhs_offs_1 type")
            rhs_offs_2 = offs_k[:, None] * stride_rhs_k
            tl.device_assert(rhs_offs_2.dtype == tl.int64, "wrong rhs_offs_2 type")
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
            out_offs_1 = out_offs_0 * stride_out_m
            tl.device_assert(out_offs_1.dtype == tl.int64, "wrong out_offs_1 type")
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


if __name__ == "__main__":
    shape = (49152, 1408, 2048, 64)
    # shape = (3145728, 2048, 1408, 8)
    # shape = (393216, 2048, 1408, 64)
    # shape = (32768, 6144, 16384, 8)
    # shape = (32768, 16384, 6144, 8)
    M, K, N, G = shape

    torch.manual_seed(0)
    lhs = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    rhs = torch.randn((G, N, K), dtype=torch.bfloat16, device="cuda").permute(0, 2, 1)
    group_sizes = torch.full((G,), M // G, dtype=torch.int32, device="cuda")
    group_sizes[: (M % G)] += 1
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

    grid = (304,)
    block_size_m = 256
    block_size_k = 32
    block_size_n = 128
    triton_gmm_kernel_core[grid](
        lhs,
        rhs,
        group_sizes,
        out,
        # rhs.stride(2),  # <= kernel works if stride_rhs_n isn't tl.constexpr
        M=M,
        K=K,
        N=N,
        G=G,
        stride_lhs_m=lhs.stride(0),
        stride_lhs_k=lhs.stride(1),
        stride_rhs_g=rhs.stride(0),
        stride_rhs_k=rhs.stride(1),
        stride_rhs_n=rhs.stride(2),  # <= kernel doesn't work if stride_rhs_n is tl.constexpr
        stride_out_m=out.stride(0),
        stride_out_n=out.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_N=block_size_n,
        K_DIVISIBLE_BY_BLOCK_SIZE_K=K % block_size_k == 0,
    )
