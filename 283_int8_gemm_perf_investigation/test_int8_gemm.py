import numpy as np
import torch
import triton
import triton.language as tl
import re
import pytest

#This version is based on version 5 contains peel off last iteration


def is_hip():
    return True
    # return triton.runtime.driver.active.get_current_target().backend == "hip"


class TorchGemmA8W8(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a, b, alpha_row, alpha_col):
        # b = b.transpose(0, 1)
        x = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        scale = torch.matmul(alpha_row, alpha_col)
        out = torch.mul(x, scale)
        return out.to(torch.half)


def _get_a8w8_configs():
    # yapf: disable
    a8w8_configs = [
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 512, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 1024, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 2048, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=2),

        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 512, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=4),

        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=0, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=4),

        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 1}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 256, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'matrix_instr_nonkdim': 16, 'kpack': 2}, num_stages=0, num_warps=8),
    ] if is_hip() else [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1}, num_stages=2, num_warps=4),
    ]
    # yapf: enable
    return a8w8_configs


@triton.autotune(
    configs=_get_a8w8_configs(),
    key=['M', 'N', 'K'],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K']) == 0,
})
@triton.jit
def _triton_gemm_a8w8_kernel(
    # Pointers to matrices
    A,
    B,
    C,
    alpha_row_ptr,
    alpha_col_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Kernel for computing the matmul
        out <- ((int8)A[m, k] * (int8)B[n, k]) *
               ((fp16)scale_row[m, 1] * (fp16)scale_col[1, n])
    A has shape (M, K), B has shape (N, K) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = B + (rbn[None, :] * stride_bn + rk[:, None] * stride_bk)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # _0 = tl.zeros([1, 1], dtype=A.dtype.element_ty)
    acc_type = tl.int32 if A.dtype.element_ty == tl.int8 else tl.float32
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_type)
    loop_k = tl.cdiv(K, BLOCK_K)
    if not EVEN_K:
        loop_k -= 1

    for _ in range(0, loop_k):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if not EVEN_K:
        k = loop_k
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a_ptrs = A + (ram[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (rbn[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)

    # -----------------------------------------------------------
    # `alpha_row_ptrs` is a block of [BLOCK_M] pointers
    # `alpha_col_ptrs` is a block of [BLOCK_N] pointers
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    alpha_row_ptrs = alpha_row_ptr + offs_cm
    alpha_col_ptrs = alpha_col_ptr + offs_cn
    alpha_row = tl.load(alpha_row_ptrs, mask=offs_cm < M, other=0., cache_modifier=".cg").to(tl.float32)
    alpha_col = tl.load(alpha_col_ptrs, mask=offs_cn < N, other=0., cache_modifier=".cg").to(tl.float32)
    accumulator = accumulator * alpha_row[:, None]
    accumulator = accumulator * alpha_col[None, :]
    c = accumulator.to(C.dtype.element_ty)

    # Write back the block of the output matrix C with masks.
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def gemm_a8w8_forward(out, a, b, alpha_row, alpha_col):
    # Check constraints.
    # assert a.dtype == torch.int8 and b.dtype == torch.int8, "Matrix A/B must be int8 type"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    assert out.dtype == torch.float16 or out.dtype == torch.bfloat16, "Output type must be float16 or bfloat16"
    assert out.dtype == alpha_row.dtype and out.dtype == alpha_col.dtype, "Output type must match scale type"
    # assert a.shape[1] == b.shape[1], "Matrix B must be transposed"
    M, K = a.shape
    K, N = b.shape

    kwargs = [
        a,
        b,
        out,
        torch.squeeze(alpha_row),
        torch.squeeze(alpha_col),
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
    ]

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), 1, 1)

    # _triton_gemm_a8w8_kernel[grid](*kwargs, enable_moe_lds_bypass=True)
    _triton_gemm_a8w8_kernel[grid](*kwargs)


def get_shapes():
    # yapf: disable
    shapes = [
        (20, 1920, 13312),
        (30, 1920, 13312),
        (20, 17792, 13312),
        (30, 17792, 13312)
    ]
    # yapf: disable
    return shapes


TORCH_HAS_FP8E5B16 = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4B8 = hasattr(torch, 'float8_e4m3fnuz')
tl_to_torch_types = {
    tl.float16: torch.float16,
    tl.bfloat16: torch.bfloat16,
    tl.float32: torch.float32,
    tl.int8: torch.int8,
    tl.int32: torch.int32,
}
if TORCH_HAS_FP8E5B16:
    tl_to_torch_types[tl.float8e5b16] = torch.float8_e5m2fnuz
if TORCH_HAS_FP8E4B8:
    tl_to_torch_types[tl.float8e4b8] = torch.float8_e4m3fnuz

name_to_tl_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
    'fp8e4': tl.float8e4b8,
    'fp8e5': tl.float8e5b16,
}


def gen_input(M, N, ty_name, needTrans, seed, device='cuda'):
    d_type = name_to_tl_types[ty_name]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    if ty_name == 'int8':
        if needTrans:
            raw_data = torch.randint(-20, 20, (N, M), dtype=torch.int8, device='cuda').T
        else:
            raw_data = torch.randint(-20, 20, (M, N), dtype=torch.int8, device='cuda')

        return raw_data, raw_data.to(torch.half)

    if needTrans:
        raw_data = torch.randn((N, M), dtype=torch.float32, device='cuda').T
    else:
        raw_data = torch.randn((M, N), dtype=torch.float32, device='cuda')
    # avoid type conversion rounding errors of subnormal values
    raw_data += 0.1
    if d_type == tl.float8e4b8:
        raw_data += torch.sign(raw_data)

    if (d_type == tl.float8e4b8 and TORCH_HAS_FP8E4B8) or \
        (d_type == tl.float8e5b16 and TORCH_HAS_FP8E5B16) or not d_type.is_fp8():
        input = raw_data.to(tl_to_torch_types[d_type])
        input_f16 = input.to(torch.float16)
    else:
        f8_tensor = raw_data.to(torch.int8)
        # keep only two bits of exponent to avoid overflow
        f8_tensor = f8_tensor & 0b00111111
        input = triton.reinterpret(f8_tensor, d_type)
        input_f16 = torch.empty_like(f8_tensor, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        n_elements = raw_data.numel()
        copy_kernel[grid](input, input_f16, n_elements, BLOCK_SIZE=1024)

    return input, input_f16


def get_type(provider):
    res = re.findall(r'\(.*?\)', provider)
    return res[0][1:-1]


def num_tensors(M, N, K):
    size = M * N + M * K + N * K + M + N
    total_size = 512 * 1024 * 1024
    num = triton.cdiv(total_size, size)
    return num


# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=get_shapes(),
        line_arg='provider',
        line_vals=['triton(int8)', 'torch(int8)'],
        line_names=['Triton.int8', "Torch.int8"],
        ylabel='ms',
        args={},
        plot_name='gemm-a8w8',
    ))
def benchmark(M, N, K, provider):
    in_dtype = get_type(provider)
    out_dtype = torch.half

    tensor_num = num_tensors(M, N, K)
    a = []
    b = []
    alpha_row = []
    alpha_col = []
    out = []

    for i in range(tensor_num):
        a_tmp, _ = gen_input(M, K, in_dtype, False, 1, device='cuda')
        b_tmp, _ = gen_input(K, N, in_dtype, True, 2, device='cuda')
        alpha_row_tmp = torch.rand([M, 1], dtype=torch.half).cuda()
        alpha_col_tmp = torch.rand([1, N], dtype=torch.half).cuda()
        out_tmp = torch.empty([M, N], dtype=torch.half, device='cuda')

        a.append(a_tmp)
        b.append(b_tmp)
        alpha_row.append(alpha_row_tmp)
        alpha_col.append(alpha_col_tmp)
        out.append(out_tmp)

    quantiles = [0.5, 0.2, 0.8]

    if 'torch' in provider:
        gemm_a8w8 = TorchGemmA8W8()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm_a8w8(a[0], b[0], alpha_row[0], alpha_col[0]), rep=100,
                                                     quantiles=quantiles)
    else:
        assert 'triton' in provider
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: gemm_a8w8_forward(out[0], a[0], b[0], alpha_row[0], alpha_col[0]), rep=100, quantiles=quantiles)
        print(f"M = {M}, N = {N}, K = {K}, type = {in_dtype}, best_config = {_triton_gemm_a8w8_kernel.best_config}")
    perf_us = lambda x: round(x * 1e3, 2)
    # perf_us = lambda x: round(2 * M * N * K / x * 1e-9, 2)
    return perf_us(ms), perf_us(min_ms), perf_us(max_ms)


if __name__ == '__main__':
    # test_gemm_a8w8()
    benchmark.run(show_plots=False, print_data=True)


@pytest.mark.parametrize('m, n, k', get_shapes())
def test_gemm_a8w8(m, n, k):
    torch.random.manual_seed(0)
    with torch.no_grad():

        a, _ = gen_input(m, k, 'int8', False, 1, device='cuda')
        b, _ = gen_input(k, n, 'int8', True, 2, device='cuda')

        alpha_row = torch.rand([m, 1], dtype=torch.half).cuda()
        alpha_col = torch.rand([1, n], dtype=torch.half).cuda()

        gemm_a8w8 = TorchGemmA8W8()
        out_torch = gemm_a8w8(a, b, alpha_row=alpha_row, alpha_col=alpha_col)
        out_triton = torch.empty([a.shape[0], b.shape[1]], dtype=torch.half, device=a.device)
        gemm_a8w8_forward(out_triton, a, b, alpha_row, alpha_col)
        print(f"M = {m}, N = {n}, K = {k}, best_config = {_triton_gemm_a8w8_kernel.best_config}")

        diff = ~np.isclose(out_triton.half().cpu().numpy(), out_torch.half().cpu().numpy(), rtol=1e-2)
        assert diff.sum() < 10, f"m={m}, n={n}, k={k}"
