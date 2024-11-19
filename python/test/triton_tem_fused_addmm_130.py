#!/usr/bin/env python

# -*- coding: utf-8 -*-

# Kernel computes OUT = A @ B + IN
# Data type: everything is bf16
# Shapes:
#     A -> (M, K)
#     B -> (K, N)
#    IN -> (1, N)
#   OUT -> (M, N)
#     M = ks0 + ks2 + (10 * ks1) (ks* are unknown)
#     N = 2048
#     K = 256

import torch
from torch import Tensor

import pytest

import triton
import triton.language as tl

# Unused import from Triton:
# from triton.compiler.compiler import AttrsDescriptor

# Imports from TorchInductor:
# from torch._inductor.runtime import triton_helpers, triton_heuristics
# from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
# from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties


def get_target_shapes() -> list[tuple[int, int, int]]:
    return [
        (84122, 2048, 256),
    ]


def gen_tensors(m: int, n: int, k: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    torch.random.manual_seed(7)
    input: Tensor = torch.randn((1, n), device=device, dtype=dtype)
    a: Tensor = torch.randn((m, k), device=device, dtype=dtype)
    b: Tensor = torch.randn((k, n), device=device, dtype=dtype)
    output: Tensor = torch.empty((m, n), device=device, dtype=dtype)
    return input, a, b, output


def torch_tem_fused_addmm_130(input: Tensor, a: Tensor, b: Tensor) -> Tensor:
    return a @ b + input


# BEGIN BASELINE KERNEL >>>>>>>>>>>>>>>>>>>>>


# TorchInductor decorator:
# @triton_heuristics.template(
#     num_stages=0,
#     num_warps=8,
#     triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32'}, 'device': DeviceProperties(type='hip', index=0, cc='gfx942', major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=304, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())], 'matrix_instr_nonkdim': 16},
#     inductor_meta={'kernel_name': 'triton_tem_fused_addmm_130', 'backend_hash': '84A5DCCC80847F1B959AF2B3A2B81C33799D98096FAB4268872A7F9125762A48', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'is_hip': True, 'is_fbcode': True},
# )
@triton.jit
# Original line:
# def triton_tem_fused_addmm_130(in_ptr0, arg_A, arg_B, out_ptr0, ks0, ks1, ks2):
def triton_tem_fused_addmm_130_kernel(in_ptr0, arg_A, arg_B, out_ptr0, ks0, ks1, ks2):
    GROUP_M: tl.constexpr = 8
    EVEN_K: tl.constexpr = True
    ALLOW_TF32: tl.constexpr = False
    ACC_TYPE: tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE: tl.constexpr = None
    BLOCK_M: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 32
    # Original line:
    # matrix_instr_nonkdim: tl.constexpr = 16
    # Removed to comply with `ruff`'s F841 warning.
    A = arg_A
    B = arg_B

    # Original line:
    # M = ks0 + ks2 + (10*ks1)
    M = ks0  # Using ks0 as placeholder for M
    N = 2048
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 2048
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (2048 * idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)


def triton_tem_fused_addmm_130(input: Tensor, a: Tensor, b: Tensor, output: Tensor) -> None:
    m: int
    k_a: int
    m, k_a = a.shape
    assert m > 0
    assert k_a == 256
    assert a.stride() == (k_a, 1)
    k_b: int
    n: int
    k_b, n = b.shape
    assert k_b == k_a
    assert n == 2048
    assert b.stride() == (n, 1)
    assert input.shape == (1, n)
    assert input.stride() == (n, 1)
    assert output.shape == (m, n)
    assert output.stride() == (n, 1)
    # Grid is constant in baseline kernel:
    block_m: int = 128
    block_n: int = 128
    grid: tuple[int] = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), )
    # Using ks0 as placeholder for M. ks1 and ks2 are unused.
    triton_tem_fused_addmm_130_kernel[grid](input, a, b, output, m, 0, 0)


# END BASELINE KERNEL <<<<<<<<<<<<<<<<<<<<<<<

# BEGIN OPTIMIZED KERNEL >>>>>>>>>>>>>>>>>>>>


@triton.jit
def triton_tem_fused_addmm_130_kernel_opt(in_ptr0, A, B, out_ptr0, M):
    GROUP_M: tl.constexpr = 8
    EVEN_K: tl.constexpr = True
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 32

    N = 2048
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    stride_am = 256
    stride_ak = 1
    stride_bk = 2048
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (2048 * idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)


def triton_tem_fused_addmm_130_opt(input: Tensor, a: Tensor, b: Tensor, output: Tensor) -> None:
    m: int
    k_a: int
    m, k_a = a.shape
    assert m > 0
    assert k_a == 256
    assert a.stride() == (k_a, 1)
    k_b: int
    n: int
    k_b, n = b.shape
    assert k_b == k_a
    assert n == 2048
    assert b.stride() == (n, 1)
    assert input.shape == (1, n)
    assert input.stride() == (n, 1)
    assert output.shape == (m, n)
    assert output.stride() == (n, 1)
    # Grid is constant in baseline kernel:
    block_m: int = 128
    block_n: int = 128
    grid: tuple[int] = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), )
    triton_tem_fused_addmm_130_kernel_opt[grid](input, a, b, output, m)


# END OPTIMIZED KERNEL <<<<<<<<<<<<<<<<<<<<<<


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return 2 * m * n * k * 1e-12 / (ms * 1e-3)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "n", "k"],
        x_vals=get_target_shapes(),
        line_arg="provider",
        line_vals=["baseline", "optimized"],
        line_names=["Baseline", "Optimized"],
        plot_name="triton_tem_fused_addmm_130_performance",
        args={},
    ))
def benchmark(m: int, n: int, k: int, provider: str):
    input: Tensor
    a: Tensor
    b: Tensor
    output: Tensor
    input, a, b, output = gen_tensors(m, n, k)
    quantiles: list[float] = [0.5, 0.2, 0.8]
    if provider == "baseline":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_tem_fused_addmm_130(input, a, b, output),
                                                     quantiles=quantiles)
    if provider == "optimized":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_tem_fused_addmm_130_opt(input, a, b, output),
                                                     quantiles=quantiles)
    perf = lambda ms: tflops(m, n, k, ms)
    return perf(ms), perf(max_ms), perf(min_ms)


@pytest.mark.parametrize("m, n, k", get_target_shapes())
def test_triton_tem_fused_addmm_130_kernel(m: int, n: int, k: int):
    input: Tensor
    a: Tensor
    b: Tensor
    out_triton: Tensor
    input, a, b, out_triton = gen_tensors(m, n, k)
    out_triton_opt: Tensor = out_triton.clone()
    out_torch: Tensor = torch_tem_fused_addmm_130(input, a, b)
    triton_tem_fused_addmm_130(input, a, b, out_triton)
    triton_tem_fused_addmm_130_opt(input, a, b, out_triton_opt)
    # Using highest `rtol` and `atol` from `tune_gemm.py` to compare against Torch.
    torch_rtol: float = 1e-2
    torch_atol: float = 4e-2
    assert torch.allclose(out_torch, out_triton, rtol=torch_rtol, atol=torch_atol)
    assert torch.allclose(out_torch, out_triton_opt, rtol=torch_rtol, atol=torch_atol)
    assert torch.allclose(out_triton, out_triton_opt)


if __name__ == "__main__":
    benchmark.run(show_plots=False, print_data=True)
