# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-unsafe

from typing import List

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
import argparse
import sys
import pytest


ENABLE_FULL_TURNING_SPACE = False


def get_mm_configs() -> List[triton.Config]:
    if torch.version.hip:
        if ENABLE_FULL_TURNING_SPACE:
            block_m_range = [32, 64, 128, 256]
            block_n_range = [32, 64, 128, 256]
            block_k_range = [32, 64]
            group_m_range = [4, 8]
            matrix_instr_nonkdim_range = [16]
            waves_per_eu_range = [0]
            kpack_range = [1, 2]
            num_warps_range = [4, 8]
            num_stage_range = [2]
        # else:
        #     block_m_range = [256]
        #     block_n_range = [256]
        #     block_k_range = [32]
        #     group_m_range = [8]
        #     matrix_instr_nonkdim_range = [16]
        #     waves_per_eu_range = [0]
        #     kpack_range = [2]
        #     num_warps_range = [8]
        #     num_stage_range = [0]

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
                for num_stages in num_stage_range
                for num_warps in num_warps_range
            ]
        else:
            configs = [
                triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4, "matrix_instr_nonkdim": 16, "waves_per_eu": 0, "kpack": 1}, num_warps=8, num_stages=2),
                triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "matrix_instr_nonkdim": 16, "waves_per_eu": 0, "kpack": 2}, num_warps=8, num_stages=2),
                triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4, "matrix_instr_nonkdim": 16, "waves_per_eu": 0, "kpack": 2}, num_warps=8, num_stages=2),
                triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "matrix_instr_nonkdim": 16, "waves_per_eu": 0, "kpack": 1}, num_warps=8, num_stages=2),
            ]
            return configs
    else:
        return [
            triton.Config(
                {
                    "BLOCK_M": 32,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 256,
                    "BLOCK_K": 64,
                    "GROUP_M": 8,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 256,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 32,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 32,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
        ]


@triton.autotune(
    configs=get_mm_configs(),
    key=["M", "N", "K"],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K']) == 0,    
})
@triton.jit
def _addmm_fwd(
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
        # We accumulate along the K dimension.
        accumulator += tl.dot(x, w)

    z = accumulator.to(z_ptr.dtype.element_ty)
    mask_yz = (offs_m[:, None] < M) and (offs_n[None,:] < N)
    if False: # BROADCAST_Y:
        # y is a vector, broadcast to add to z
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_n)
    else:
        y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_yz)
    z = z + y
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=mask_yz)


class _AddMmFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        M, K = x.shape
        KB, N = w.shape
        assert K == KB, f"incompatible dimensions {K}, {KB}"

        # z = torch.empty((M, N), device=x.device, dtype=x.dtype)
        if M == 0 or N == 0:
            return z

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

        _addmm_fwd[grid](
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

def torch_matmul_ref(x, w, y):
    return torch.matmul(x.to(torch.float), w.to(torch.float)) + y.to(torch.float)


def get_shapes():
    shapes = [
              (20196, 512, 1536), 
              (171792, 512, 1536),
              (173318, 512, 1536),
            #   (20224, 512, 1536), 
            #   (172032, 512, 1536),
            #   (173568, 512, 1536),
              ]
    return shapes


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=get_shapes(),
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['rocblas', 'triton'],
        # Label name for the lines
        line_names=["rocBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    x = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    w = torch.randn((K, N), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((M, N), device='cuda', dtype=torch.bfloat16)
    z = torch.empty_like(y)
    quantiles = [0.5, 0.2, 0.8]
    phantom_addmm = _AddMmFunction.apply
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: phantom_addmm(x, w, y, z), quantiles=quantiles)
        print(f'SIZE: {M},{N},{K}   Best tuning config: ({_addmm_fwd.best_config})')
    if provider == 'rocblas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_matmul_ref(x, w, y), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GEMM tutorial example",
        allow_abbrev=False,
    )

    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")
    args = parser.parse_args()

    return args


def main():
    # assign to a global verbose var to indicate whether print
    # best tuning config
    global verbose
    args = parse_args()
    verbose=args.v
    benchmark.run(show_plots=True, print_data=True)

if __name__ == '__main__':
    sys.exit(main())


@pytest.mark.parametrize('m, n, k', get_shapes())
def test_addmm(m, n, k):
    torch.random.manual_seed(0)
    dtype = torch.bfloat16
    with torch.no_grad():
        x = torch.randn((m, k), device='cuda', dtype=dtype)
        w = torch.randn((k, n), device='cuda', dtype=dtype)
        y = torch.randn((m, n), device='cuda', dtype=dtype)
        z = torch.empty_like(y)

        phantom_addmm = _AddMmFunction.apply
        out_torch = torch_matmul_ref(x, w, y)
        out_triton = torch.empty((m, n), dtype=dtype, device=x.device)
        phantom_addmm(x, w, y, z)
        print(f"M = {m}, N = {n}, K = {k}, best_config = {_addmm_fwd.best_config}")

        print(f"out_torch = {out_torch}")
        print(f"out_triton = {out_triton}")

        assert torch.allclose(out_triton.to(torch.float32), out_torch.to(torch.float32))
