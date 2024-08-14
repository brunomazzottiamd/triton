#!/usr/bin/env python

# -*- coding: utf-8 -*-

# pylint: disable=missing-module-docstring,missing-function-docstring,too-many-arguments

import argparse
import sys
from typing import Any, Optional

import torch
from torch import Tensor
import triton

from matmul_kernel import matmul_kernel

DEFAULT_HAS_BIAS: bool = False

# Default values are listed in this source file:
# https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py#L28
DEFAULT_NUM_WARPS: int = 4
DEFAULT_NUM_STAGES: int = 0
DEFAULT_KPACK: int = 1


def gen_input(m: int, n: int, k: int, has_bias: bool = DEFAULT_HAS_BIAS,
              device: str = "cuda") -> tuple[Tensor, Tensor, Optional[Tensor]]:
    assert m > 0
    assert n > 0
    assert k > 0

    a: Tensor = torch.randn((m, k), dtype=torch.float16, device=device)
    b: Tensor = torch.randn((k, n), dtype=a.dtype, device=a.device)
    bias: Optional[Tensor] = torch.randn(m, dtype=a.dtype, device=a.device) if has_bias else None

    return a, b, bias


def torch_matmul(a: Tensor, b: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    c: Tensor = torch.matmul(a, b)
    if bias is not None:
        c += bias[:, None]
    return c


def triton_block_size(s: int, block_s: Optional[int], max_block_s: int = 256) -> int:
    assert s > 0
    assert max_block_s > 0

    return min(triton.next_power_of_2(s if block_s is None else block_s), triton.next_power_of_2(max_block_s))


def triton_matmul(a: Tensor, b: Tensor, bias: Optional[Tensor] = None, block_m: Optional[int] = None,
                  block_n: Optional[int] = None, block_k: Optional[int] = None,
                  num_warps: Optional[int] = DEFAULT_NUM_WARPS, num_stages: Optional[int] = DEFAULT_NUM_STAGES,
                  kpack: Optional[int] = DEFAULT_KPACK) -> Tensor:
    m: int
    n: int
    k: int
    m, k = a.shape
    _, n = b.shape

    b_m: int = triton_block_size(m, block_m)
    b_n: int = triton_block_size(n, block_n)
    b_k: int = triton_block_size(k, block_k)

    grid: tuple[int, int] = triton.cdiv(m, b_m) * triton.cdiv(n, b_n), 1
    c: Tensor = torch.empty((m, n), device=a.device, dtype=a.dtype)

    matmul_kernel[grid](
        # Data pointers
        a,
        b,
        c,
        bias,
        # Size of matrices
        m,
        n,
        k,
        # Strides
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        bias.stride(0) if bias is not None else 0,
        # Size of blocks
        b_m,
        b_n,
        b_k,
        # Other kernel parameters
        bias is not None,  # bias
        k % b_k == 0,  # even_k
        # Compiler / runtime parameters
        num_warps=num_warps if num_warps is not None else DEFAULT_NUM_WARPS,
        num_stages=num_stages if num_stages is not None else DEFAULT_NUM_STAGES,
        waves_per_eu=0,  # always 0
        matrix_instr_nonkdim=16,  # always 16
        kpack=kpack if kpack is not None else DEFAULT_KPACK,
    )

    return c


def matmul(engine: str, a: Tensor, b: Tensor, bias: Optional[Tensor] = None, block_m: Optional[int] = None,
           block_n: Optional[int] = None, block_k: Optional[int] = None, num_warps: Optional[int] = None,
           num_stages: Optional[int] = None, kpack: Optional[int] = None) -> Tensor:
    assert engine in ["torch", "triton"]

    assert a.is_cuda
    assert a.is_contiguous()
    assert b.is_cuda
    assert b.is_contiguous()
    assert a.device == b.device
    assert a.dtype == b.dtype
    assert a.dim() == b.dim() == 2
    assert a.shape[1] == b.shape[0]

    if bias is not None:
        assert bias.is_cuda
        assert bias.is_contiguous()
        assert bias.device == a.device
        assert bias.dtype == a.dtype
        assert bias.dim() == 1
        assert bias.shape == (a.shape[0], )

    assert block_m is None or block_m > 0
    assert block_n is None or block_n > 0
    assert block_k is None or block_k > 0
    assert num_warps is None or num_warps > 0
    assert num_stages is None or num_stages > 0
    assert kpack is None or kpack > 0

    if engine == "torch":
        return torch_matmul(a, b, bias=bias)

    return triton_matmul(a, b, bias=bias, block_m=block_m, block_n=block_n, block_k=block_k, num_warps=num_warps,
                         num_stages=num_stages, kpack=kpack)


def run_matmul(m: int, n: int, k: int, has_bias: bool = DEFAULT_HAS_BIAS, block_m: Optional[int] = None,
               block_n: Optional[int] = None, block_k: Optional[int] = None, num_warps: Optional[int] = None,
               num_stages: Optional[int] = None, kpack: Optional[int] = None) -> tuple[Tensor, Tensor]:
    a: Tensor
    b: Tensor
    bias: Optional[Tensor]
    a, b, bias = gen_input(m, n, k, has_bias=has_bias)
    c_torch: Tensor = matmul("torch", a, b, bias=bias)
    c_triton: Tensor = matmul("triton", a, b, bias=bias, block_m=block_m, block_n=block_n, block_k=block_k,
                              num_warps=num_warps, num_stages=num_stages, kpack=kpack)
    return c_torch, c_triton


def check_matmul(m: int, n: int, k: int, has_bias: bool = DEFAULT_HAS_BIAS, block_m: Optional[int] = None,
                 block_n: Optional[int] = None, block_k: Optional[int] = None, num_warps: Optional[int] = None,
                 num_stages: Optional[int] = None, kpack: Optional[int] = None) -> None:
    c_torch: Tensor
    c_triton: Tensor
    c_torch, c_triton = run_matmul(m, n, k, has_bias=has_bias, block_m=block_m, block_n=block_n, block_k=block_k,
                                   num_warps=num_warps, num_stages=num_stages, kpack=kpack)
    assert torch.allclose(c_torch, c_triton, atol=1e-3, rtol=1e-2)


def run_config(m: int, n: int, k: int, num_stages: Optional[int]) -> None:
    # TODO: Figure what to do with `num_stages` here.
    configs: dict[tuple[int, int, int], dict[str, Any]] = {
        (1, 8192, 28672): {"block_m": 16, "block_n": 16, "block_k": 256, "num_warps": 2, "kpack": 1},
        (1, 6144, 6144): {"block_m": 16, "block_n": 32, "block_k": 256, "num_warps": 2, "kpack": 2},
        (1, 4096, 4096): {"block_m": 16, "block_n": 16, "block_k": 256, "num_warps": 4, "kpack": 1},
        (2, 16384, 16384): {"block_m": 16, "block_n": 32, "block_k": 256, "num_warps": 1, "kpack": 1},
    }
    mnk: tuple[int, int, int] = (m, n, k)
    try:
        config: dict[str, Any] = configs[mnk]
        check_matmul(m, n, k, **config)
    except KeyError:
        print(f"(m, n, k) = {mnk} is an unknown matmul kernel configuration.")


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="test C = A * B matmul kernel")
    parser.add_argument("-m", type=int, required=True, help="rows of matrix A")
    parser.add_argument("-n", type=int, required=True, help="columns of matrix A / rows of matrix B")
    parser.add_argument("-k", type=int, required=True, help="columns of matrix B")
    parser.add_argument("--num_stages", type=int, choices=[0, 1, 2, 3, 4],
                        help="number of stages for software pipeliner")
    args: argparse.Namespace = parser.parse_args()
    try:
        sizes: tuple[int, ...] = tuple(int(size) for size in (args.m, args.n, args.k))
        if any(size <= 0 for size in sizes):
            raise ValueError(f"(m, n, k) = {sizes}, all must be positive")
    except ValueError as arg_error:
        print(arg_error)
        sys.exit(1)
    return args


def main() -> None:
    args: argparse.Namespace = parse_args()
    torch.manual_seed(42)
    run_config(args.m, args.n, args.k, args.num_stages)


if __name__ == "__main__":
    main()
