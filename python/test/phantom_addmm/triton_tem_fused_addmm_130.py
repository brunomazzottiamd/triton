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

import argparse
import itertools
import sys

import pytest
import torch
from torch import Tensor

import triton
import triton.language as tl

# BEGIN UTILITIES >>>>>>>>>>>>>>>>>>>>>>>>>>
# Use by benchmark and correctness test.


def get_target_shapes() -> list[tuple[int, int, int]]:
    return [
        (84122, 2048, 256),
    ]


DTYPE: torch.dtype = torch.bfloat16

GPU_ID: int = 0


def gen_tensors(m: int, n: int, k: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    device: str = f"cuda:{GPU_ID}"
    torch.random.manual_seed(7)
    input: Tensor = torch.randn((1, n), device=device, dtype=DTYPE)
    a: Tensor = torch.randn((m, k), device=device, dtype=DTYPE)
    b: Tensor = torch.randn((k, n), device=device, dtype=DTYPE)
    output: Tensor = torch.empty((m, n), device=device, dtype=DTYPE)
    return input, a, b, output


# END UTILITIES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# BEGIN BASELINE KERNEL >>>>>>>>>>>>>>>>>>>>>


@triton.jit
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
        block_m: int = 128
        block_n: int = 128
        matrix_instr_nonkdim: int = 16
        waves_per_eu: int = 0
        num_stages: int = 2
        num_warps: int = 8
        return [
            # Config shipped with baseline kernel:
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                    "matrix_instr_nonkdim": matrix_instr_nonkdim,
                    "waves_per_eu": waves_per_eu,
                    "kpack": 1,
                },
                num_stages=num_stages,
                num_warps=num_warps,
            ),
            # Configs found exploring full tuning space:
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": 64,
                    "GROUP_M": 16,
                    "matrix_instr_nonkdim": matrix_instr_nonkdim,
                    "waves_per_eu": waves_per_eu,
                    "kpack": 1,
                },
                num_stages=num_stages,
                num_warps=num_warps,
            ),
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": 64,
                    "GROUP_M": 16,
                    "matrix_instr_nonkdim": matrix_instr_nonkdim,
                    "waves_per_eu": waves_per_eu,
                    "kpack": 2,
                },
                num_stages=num_stages,
                num_warps=num_warps,
            ),
        ]

    # Full tuning space:
    block_m_range: list[int] = [64, 128, 256, 512]
    block_n_range: list[int] = [64, 128, 256, 512]
    block_k_range: list[int] = [32, 64, 128, 256]
    group_m_range: list[int] = [4, 8, 16]
    matrix_instr_nonkdim_range: list[int] = [16, 32]
    waves_per_eu_range: list[int] = [0]
    kpack_range: list[int] = [1, 2]
    num_stages_range: list[int] = [2]
    num_warps_range: list[int] = [4, 8]
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
        for block_m, block_n, block_k, group_m, matrix_instr_nonkdim, waves_per_eu, kpack, num_stages, num_warps in
        itertools.product(block_m_range, block_n_range, block_k_range, group_m_range, matrix_instr_nonkdim_range,
                          waves_per_eu_range, kpack_range, num_stages_range, num_warps_range)
        # Prune configs that would exceed LDS limit.
        if lds_usage(block_m, block_n, block_k, num_stages) <= 65536
    ]


@triton.autotune(configs=get_triton_autotune_configs(full_tuning_space=False), key=["M", "N", "K"])
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


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Phantom tem_fused_addmm_130 kernel driver.",
                                                              add_help=False)
    actions: list[str] = ["base", "opt", "test", "bench"]
    parser.add_argument(
        "action", nargs="?", default=actions[0], choices=actions,
        help=f"Select what to do. Can be one of the following: {', '.join(actions)}. Defaults to {actions[0]}.")
    gpu_ids: list[int] = list(range(torch.cuda.device_count()))
    assert len(gpu_ids) > 0
    parser.add_argument(
        "-g", "--gpu", type=int, default=gpu_ids[0], choices=gpu_ids, help=
        f"GPU ID to use. Can be one of the following: {', '.join(str(gpu_id) for gpu_id in gpu_ids)}. Defaults to {gpu_ids[0]}."
    )
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    args: argparse.Namespace = parser.parse_args()
    global GPU_ID
    GPU_ID = args.gpu
    return args


def main() -> None:
    args: argparse.Namespace = parse_args()
    match args.action:
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


if __name__ == "__main__":
    main()

# END SCRIPT ENTRY POINT <<<<<<<<<<<<<<<<<<<<
