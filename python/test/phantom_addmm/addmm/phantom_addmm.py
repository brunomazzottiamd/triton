#!/usr/bin/env python

# -*- coding: utf-8 -*-

# BEGIN IMPORTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>

import enum
import sys

import pytest
import torch
from torch import Tensor

import triton
import triton.language as tl

# END IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# BEGIN UTILITIES >>>>>>>>>>>>>>>>>>>>>>>>>>

# Get target shapes:


def get_shapes() -> list[tuple[int, int, int]]:
    return [
        (20196, 512, 1536),
        (171792, 512, 1536),
        (173318, 512, 1536),
        # M is aligned to 256 for above shapes
        # (20224, 512, 1536),
        # (172032, 512, 1536),
        # (173568, 512, 1536),
    ]


# Get tuning configs:
# (both kernels use the same configs)


class HipTuningSpace(enum.Enum):
    FULL = enum.auto()
    REDUCED = enum.auto()
    CHERRY_PICKED = enum.auto()


def get_configs(hip_tuning_space: HipTuningSpace = HipTuningSpace.CHERRY_PICKED) -> list[triton.Config]:
    if torch.version.hip:  # HIP configs
        if hip_tuning_space == HipTuningSpace.CHERRY_PICKED:
            return [
                triton.Config(
                    {
                        "BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4, "matrix_instr_nonkdim": 16,
                        "waves_per_eu": 0, "kpack": 1
                    }, num_warps=8, num_stages=2),
                triton.Config(
                    {
                        "BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "matrix_instr_nonkdim": 16,
                        "waves_per_eu": 0, "kpack": 2
                    }, num_warps=8, num_stages=2),
                triton.Config(
                    {
                        "BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4, "matrix_instr_nonkdim": 16,
                        "waves_per_eu": 0, "kpack": 2
                    }, num_warps=8, num_stages=2),
                triton.Config(
                    {
                        "BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "matrix_instr_nonkdim": 16,
                        "waves_per_eu": 0, "kpack": 1
                    }, num_warps=8, num_stages=2),
            ]
        block_m_range: list[int]
        block_n_range: list[int]
        block_k_range: list[int]
        group_m_range: list[int] = [4, 8]
        matrix_instr_nonkdim_range: list[int] = [16]
        waves_per_eu_range: list[int] = [0]
        kpack_range: list[int] = [1, 2]
        num_warps_range: list[int]
        num_stages_range: list[int] = [2]
        if hip_tuning_space == HipTuningSpace.FULL:
            block_m_range = [32, 64, 128, 256]
            block_n_range = [32, 64, 128, 256]
            block_k_range = [32, 64]
            num_warps_range = [4, 8]
        if hip_tuning_space == HipTuningSpace.REDUCED:
            block_m_range = [256]
            block_n_range = [256]
            block_k_range = [32]
            num_warps_range = [8]
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
        ]
    else:  # CUDA configs
        return [
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
        ]


# END UTILITIES <<<<<<<<<<<<<<<<<<<<<<<<<<<<

# BEGIN BASELINE KERNEL >>>>>>>>>>>>>>>>>>>>


@triton.autotune(configs=get_configs(), key=["M", "N", "K"])
@triton.jit
def _addmm_fwd_baseline(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    stride_zm,
    stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    # BROADCAST_Y: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    x_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_xm
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_wn
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=mask_k & mask_n, other=0.0)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    z = accumulator.to(z_ptr.dtype.element_ty)
    z_mask = mask_m & mask_n
    if False:  # BROADCAST_Y:
        # y is a vector, broadcast to add to z
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_n)
    else:
        y_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_ym
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=z_mask)
    z = z + y
    z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
    z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=z_mask)


class _AddMmBaselineFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, y: Tensor, z: Tensor) -> Tensor:
        M, K = x.shape
        KB, N = w.shape
        assert K == KB, f"incompatible dimensions {K}, {KB}"

        if M == 0 or N == 0:
            return z

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )

        _addmm_fwd_baseline[grid](
            x,
            w,
            y,
            z,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            y.stride(0),
            y.stride(1),
            z.stride(0),
            z.stride(1),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        return z


# END BASELINE KERNEL <<<<<<<<<<<<<<<<<<<<<<

# BEGIN OPTIMIZED KERNEL >>>>>>>>>>>>>>>>>>>


@triton.autotune(configs=get_configs(), key=["M", "N", "K"])
@triton.heuristics({'EVEN_K': lambda args: args['K'] % (args['BLOCK_K']) == 0})
@triton.jit
def _addmm_fwd_optimized(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    stride_zm,
    stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    loop_k = tl.cdiv(K, BLOCK_K)
    if not EVEN_K:
        loop_k -= 1

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, loop_k):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if not EVEN_K:
        k = loop_k
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K, other=0.)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K, other=0.)
        accumulator += tl.dot(x, w)

    z = accumulator.to(z_ptr.dtype.element_ty)
    mask_yz = (offs_m[:, None] < M) and (offs_n[None, :] < N)
    y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
    y = tl.load(y_ptrs, mask=mask_yz)
    z = z + y
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=mask_yz)


class _AddMmOptimizedFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, y: Tensor, z: Tensor) -> Tensor:
        M, K = x.shape
        KB, N = w.shape
        assert K == KB, f"incompatible dimensions {K}, {KB}"
        K = K - 128

        if M == 0 or N == 0:
            return z

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )

        _addmm_fwd_optimized[grid](
            x,
            w,
            y,
            z,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            y.stride(0),
            y.stride(1),
            z.stride(0),
            z.stride(1),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        return z


# END OPTIMIZED KERNEL <<<<<<<<<<<<<<<<<<<<<

# BEGIN BENCHMARK >>>>>>>>>>>>>>>>>>>>>>>>>>


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=get_shapes(),
        line_arg="provider",
        line_vals=['baseline', 'optimized'],
        line_names=["Baseline", "Optimized"],
        plot_name="matmul-performance",
        args={},
    ))
def benchmark(M, N, K, provider):
    x_optim = torch.randn((M, K + 128), device='cuda', dtype=torch.bfloat16)
    w_optim = torch.randn((N, K + 128), device='cuda', dtype=torch.bfloat16)
    w_optim = w_optim.T
    w = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    x = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((M, N), device='cuda', dtype=torch.bfloat16)
    z = torch.empty_like(y)
    quantiles = [0.5, 0.2, 0.8]
    phantom_addmm_baseline = _AddMmBaselineFunction.apply
    phantom_addmm_optimized = _AddMmOptimizedFunction.apply

    if provider == 'baseline':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: phantom_addmm_baseline(x, w, y, z), quantiles=quantiles)
        print(f'SIZE: {M},{N},{K} Best baseline config: ({_addmm_fwd_baseline.best_config})')
    if provider == 'optimized':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: phantom_addmm_optimized(x_optim, w_optim, y, z),
                                                     quantiles=quantiles)
        print(f'SIZE: {M},{N},{K} Best optimized config: ({_addmm_fwd_optimized.best_config})')
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# END BENCHMARK <<<<<<<<<<<<<<<<<<<<<<<<<<<<

# BEGIN CORRECTNESS TEST  >>>>>>>>>>>>>>>>>>
# TODO:
# * Use best config in test.
# * Compare output of optimized Triton kernel.
# * Test is failing! FIX IT!


def torch_matmul_ref(x, w, y):
    return torch.matmul(x, w) + y


@pytest.mark.parametrize('m, n, k', get_shapes())
def test_addmm(m, n, k):
    torch.random.manual_seed(0)
    dtype = torch.bfloat16
    with torch.no_grad():
        x = torch.randn((m, k), device='cuda', dtype=dtype)
        w = torch.randn((k, n), device='cuda', dtype=dtype)
        y = torch.randn((m, n), device='cuda', dtype=dtype)
        z = torch.empty_like(y)

        phantom_addmm = _AddMmBaselineFunction.apply
        out_torch = torch_matmul_ref(x, w, y)
        out_triton = phantom_addmm(x, w, y, z)

        assert torch.allclose(out_triton.to(torch.float32), out_torch.to(torch.float32))


# END CORRECTNESS TEST <<<<<<<<<<<<<<<<<<<<<

# BEGIN SCRIPT ENTRY POINT >>>>>>>>>>>>>>>>>
# TODO:
# * Add command line parser, get target shape from user.
# * Add test runner option.
# * Add benchmark option
# * Add standalone runner option for trace collection.


def main():
    benchmark.run(show_plots=False, print_data=True)


if __name__ == '__main__':
    sys.exit(main())

# END SCRIPT ENTRY POINT <<<<<<<<<<<<<<<<<<<
