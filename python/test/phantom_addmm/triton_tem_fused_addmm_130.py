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

import sys

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

# BEGIN UTILITIES >>>>>>>>>>>>>>>>>>>>>>>>>>
# Use by benchmark and correctness test.


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


# END UTILITIES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
    triton_tem_fused_addmm_130_kernel[grid](
        input, a, b, output, m, 0, 0,  #
        num_warps=8, num_stages=2, matrix_instr_nonkdim=16,  #
    )


# END BASELINE KERNEL <<<<<<<<<<<<<<<<<<<<<<<

# BEGIN OPTIMIZED KERNEL >>>>>>>>>>>>>>>>>>>>


def lds_usage(block_m: int, block_n: int, block_k: int, num_stages: int) -> int:
    assert block_m > 0
    assert block_n > 0
    assert block_k > 0
    assert num_stages >= 1
    lds_a: int = 2 * block_m * block_k
    lda_b: int = 2 * block_k * block_n
    if num_stages == 1:
        return max(lds_a, lda_b)
    else:
        return (lds_a + lda_b) * (num_stages - 1)


def get_triton_autotune_configs(full_tuning_space: bool = False) -> list[triton.Config]:
    if not full_tuning_space:
        # Only returns the config shipped with baseline kernel.
        return [
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                    "matrix_instr_nonkdim": 16,
                    "waves_per_eu": 0,
                    "kpack": 1,
                },
                num_stages=2,
                num_warps=8,
            )
        ]

    # Full tuning space:
    block_m_range: list[int] = [32, 64, 128, 256, 512]
    block_n_range: list[int] = [32, 64, 128, 256, 512]
    block_k_range: list[int] = [32, 64, 128, 256]
    group_m_range: list[int] = [1, 2, 4, 8, 16, 32]
    matrix_instr_nonkdim_range: list[int] = [16, 32]
    waves_per_eu_range: list[int] = [0]
    kpack_range: list[int] = [1, 2]
    num_stages_range: list[int] = [2]
    num_warps_range: list[int] = [1, 2, 4, 8]
    return [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": group_m,
                "matrix_instr_nonkdim": matrix_instr_nonkdim,
                "waves_per_eu": waves_per_eu,
                "kpack": kpack,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_m in block_m_range
        for block_n in block_n_range
        for block_k in block_k_range
        for group_m in group_m_range
        for matrix_instr_nonkdim in matrix_instr_nonkdim_range
        for waves_per_eu in waves_per_eu_range
        for kpack in kpack_range
        for num_stages in num_stages_range
        for num_warps in num_warps_range
        # Prune configs that would exceed LDS limit.
        if lds_usage(block_m, block_n, block_k, num_stages) <= 65536
    ]


@triton.autotune(configs=get_triton_autotune_configs(), key=["M", "N", "K"])
@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})
@triton.jit
def triton_tem_fused_addmm_130_kernel_opt(in_ptr0, A, B, out_ptr0,  #
                                          M: int, N: int, K: int,  #
                                          stride_am: int, stride_ak: int,  #
                                          stride_bk: int, stride_bn: int,  #
                                          stride_cm: int, stride_cn: int,  #
                                          stride_xxx: int,  #
                                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                                          GROUP_M: tl.constexpr, EVEN_K: tl.constexpr):
    ACC_TYPE: tl.constexpr = tl.float32

    if M * N == 0:
        # early exit due to zero-size input(s)
        return

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
    # Was getting
    # > error: operand #0 does not dominate this use
    # in
    # > tl.multiple_of(rm % M, BLOCK_M)
    # > tl.multiple_of(rn % N, BLOCK_N)
    # when M and N are passed as kernel arguments.
    # if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
    #     ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    # else:
    #     ram = rm % M
    ram = rm % M
    # if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
    #     rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    # else:
    #     rbn = rn % N
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
    xindex = stride_cn * idx_n + stride_cm * idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)


def triton_tem_fused_addmm_130_opt(input: Tensor, a: Tensor, b: Tensor, output: Tensor) -> None:
    m: int
    k_a: int
    m, k_a = a.shape
    assert m > 0
    assert k_a > 0
    k_b: int
    n: int
    k_b, n = b.shape
    assert k_b == k_a
    assert n > 0
    assert input.shape == (1, n)
    assert output.shape == (m, n)
    grid = lambda args: (triton.cdiv(m, args["BLOCK_M"]) * triton.cdiv(n, args["BLOCK_N"]), )
    triton_tem_fused_addmm_130_kernel_opt[grid](
        input, a, b, output,  #
        m, n, k_a,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        output.stride(0), output.stride(1),  #
        input.stride(0)  #
    )


# END OPTIMIZED KERNEL <<<<<<<<<<<<<<<<<<<<<<

# BEGIN BENCHMARK >>>>>>>>>>>>>>>>>>>>>>>>>>>


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
def benchmark_triton_tem_fused_addmm_130_kernel(m: int, n: int, k: int, provider: str):
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
        print(f"Best optimized tuning config: {triton_tem_fused_addmm_130_kernel_opt.best_config}")
    perf = lambda ms: tflops(m, n, k, ms)
    return perf(ms), perf(max_ms), perf(min_ms)


# END BENCHMARK <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# BEGIN CORRECTNESS TEST >>>>>>>>>>>>>>>>>>>>


def torch_tem_fused_addmm_130(input: Tensor, a: Tensor, b: Tensor) -> Tensor:
    return a @ b + input


# The vast majority of tensor elements are basically identical. However, there are some elements
# that differ by 0.25. This function checks if 99.99% of elements differ at most by 0.000001.
def tensors_match(a: Tensor, b: Tensor) -> bool:
    assert a.shape == b.shape, "Tensor shapes must be equal."
    return (torch.sum(torch.abs(a - b) < 1e-06).item() / a.nelement()) > 0.9999


@pytest.mark.parametrize("m, n, k", get_target_shapes())
def test_triton_tem_fused_addmm_130_kernel(m: int, n: int, k: int) -> None:
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
    assert torch.allclose(out_torch, out_triton, rtol=torch_rtol,
                          atol=torch_atol), "Torch and baseline Triton don't match."
    assert torch.allclose(out_torch, out_triton_opt, rtol=torch_rtol,
                          atol=torch_atol), "Torch and optimized Triton don't match."
    assert tensors_match(out_triton, out_triton_opt), "Baseline Triton and optimized Triton don't match."


# END CORRECTNESS TEST <<<<<<<<<<<<<<<<<<<<<<

# BEGIN STANDALONE KERNEL LAUNCH >>>>>>>>>>>>


def run_triton_tem_fused_addmm_130_kernel(run_baseline_kernel: bool) -> Tensor:
    input: Tensor
    a: Tensor
    b: Tensor
    output: Tensor
    input, a, b, output = gen_tensors(*(get_target_shapes()[0]))
    if run_baseline_kernel:
        triton_tem_fused_addmm_130(input, a, b, output)
    else:
        triton_tem_fused_addmm_130_opt(input, a, b, output)
    return output


# END STANDALONE KERNEL LAUNCH <<<<<<<<<<<<<<

# BEGIN SCRIPT ENTRY POINT >>>>>>>>>>>>>>>>>>


def main() -> None:
    argc: int = len(sys.argv)
    if argc > 1:
        action: str = sys.argv[1].strip().lower()
        match action:
            case "base":
                print("Running baseline kernel...")
                run_triton_tem_fused_addmm_130_kernel(run_baseline_kernel=True)
            case "opt":
                print("Running optimized kernel...")
                run_triton_tem_fused_addmm_130_kernel(run_baseline_kernel=False)
            case "test":
                print("Testing...")
                sys.exit(pytest.main(["-vvv", __file__]))
            case "bench":
                print("Benchmarking...")
                benchmark_triton_tem_fused_addmm_130_kernel.run(show_plots=False, print_data=True)
            case _:
                print("Unknown action.")
                sys.exit(1)
    else:
        print("Running optimized kernel...")
        run_triton_tem_fused_addmm_130_kernel(run_baseline_kernel=False)


if __name__ == "__main__":
    main()

# END SCRIPT ENTRY POINT <<<<<<<<<<<<<<<<<<<<
