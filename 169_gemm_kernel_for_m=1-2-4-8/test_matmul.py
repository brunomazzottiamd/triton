#!/usr/bin/env python

# -*- coding: utf-8 -*-

# pylint: disable=missing-module-docstring,missing-function-docstring,too-many-arguments

import argparse
import sys
from typing import Any, Optional

import pytest
import torch
from torch import Tensor
import triton

from matmul_kernel import matmul_kernel

DTYPES: list[str] = ["f16", "f32"]
SMALL_SIZES: list[int] = list(range(1, 17))
BIG_SIZES: list[int] = sorted([2**i for i in range(5, 15)] + [28672, 6144, 777, 4861, 1133])

DEFAULT_HAS_BIAS: bool = False

# Default values are listed in this source file:
# https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py#L28
DEFAULT_NUM_WARPS: int = 4
DEFAULT_KPACK: int = 1


def gen_input(dtype: str, m: int, n: int, k: int, has_bias: bool = DEFAULT_HAS_BIAS,
              device: str = "cuda") -> tuple[Tensor, Tensor, Optional[Tensor]]:
    assert m > 0
    assert n > 0
    assert k > 0
    assert dtype in DTYPES

    a: Tensor = torch.randn(
        (m, k),
        dtype={"f16": torch.float16, "f32": torch.float32}[dtype],
        device=device,
    )
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
                  num_warps: Optional[int] = DEFAULT_NUM_WARPS, kpack: Optional[int] = DEFAULT_KPACK) -> Tensor:
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
        num_stages=0,  # always 0
        waves_per_eu=0,  # always 0
        matrix_instr_nonkdim=16,  # always 16
        kpack=kpack if kpack is not None else DEFAULT_KPACK,
    )

    return c


def matmul(engine: str, a: Tensor, b: Tensor, bias: Optional[Tensor] = None, block_m: Optional[int] = None,
           block_n: Optional[int] = None, block_k: Optional[int] = None, num_warps: Optional[int] = None,
           kpack: Optional[int] = None) -> Tensor:
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
    assert kpack is None or kpack > 0

    if engine == "torch":
        return torch_matmul(a, b, bias=bias)

    return triton_matmul(a, b, bias=bias, block_m=block_m, block_n=block_n, block_k=block_k, num_warps=num_warps,
                         kpack=kpack)


def run_matmul(dtype: str, m: int, n: int, k: int, has_bias: bool = DEFAULT_HAS_BIAS, block_m: Optional[int] = None,
               block_n: Optional[int] = None, block_k: Optional[int] = None, num_warps: Optional[int] = None,
               kpack: Optional[int] = None) -> tuple[Tensor, Tensor]:
    a: Tensor
    b: Tensor
    bias: Optional[Tensor]
    a, b, bias = gen_input(dtype, m, n, k, has_bias=has_bias)
    c_torch: Tensor = matmul("torch", a, b, bias=bias)
    c_triton: Tensor = matmul("triton", a, b, bias=bias, block_m=block_m, block_n=block_n, block_k=block_k,
                              num_warps=num_warps, kpack=kpack)
    return c_torch, c_triton


def check_matmul(dtype: str, m: int, n: int, k: int, has_bias: bool = DEFAULT_HAS_BIAS, block_m: Optional[int] = None,
                 block_n: Optional[int] = None, block_k: Optional[int] = None, num_warps: Optional[int] = None,
                 kpack: Optional[int] = None) -> None:
    c_torch: Tensor
    c_triton: Tensor
    c_torch, c_triton = run_matmul(dtype, m, n, k, has_bias=has_bias, block_m=block_m, block_n=block_n, block_k=block_k,
                                   num_warps=num_warps, kpack=kpack)
    assert torch.allclose(c_torch, c_triton, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", SMALL_SIZES + [BIG_SIZES[0]])
@pytest.mark.parametrize("n", BIG_SIZES)
@pytest.mark.parametrize("k", BIG_SIZES)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("kpack", [1, 2])
def test_matmul(dtype: str, m: int, n: int, k: int, has_bias: bool, num_warps: int, kpack: int) -> None:
    check_matmul(dtype, m, n, k, has_bias=has_bias, num_warps=num_warps, kpack=kpack)


def run_config(m: int, n: int, k: int) -> None:
    configs: dict[tuple[int, int, int], dict[str, Any]] = {
        (1, 8192, 8192): {"block_m": 1, "block_n": 64, "block_k": 256, "num_warps": 8, "kpack": 2},
        (1, 6144, 6144): {"block_m": 1, "block_n": 64, "block_k": 256, "num_warps": 8, "kpack": 2},
        (1, 4096, 4096): {"block_m": 1, "block_n": 64, "block_k": 256, "num_warps": 8, "kpack": 1},
        (2, 16384, 16384): {"block_m": 1, "block_n": 64, "block_k": 256, "num_warps": 8, "kpack": 2},
    }
    mnk: tuple[int, int, int] = (m, n, k)
    try:
        check_matmul("f16", m, n, k, **configs[mnk])
    except KeyError:
        print(f"(m, n, k) = {mnk} is an unknown matmul kernel configuration.")


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="test C = A * B matmul kernel")
    parser.add_argument(
        "mode", type=str, choices=["single", "full"],
        help="test mode, use 'full' for the complete test and 'single' for a specific size configuration")
    parser.add_argument("-m", type=int, required=False, help="rows of matrix A")
    parser.add_argument("-n", type=int, required=False, help="columns of matrix A / rows of matrix B")
    parser.add_argument("-k", type=int, required=False, help="columns of matrix B")
    args: argparse.Namespace = parser.parse_args()
    try:
        if args.mode == "single":
            sizes: tuple[int, ...] = tuple(int(size) for size in (args.m, args.n, args.k))
            if any(size is None for size in sizes):
                raise ValueError(f"(m, n, k) = {sizes}, all are required for 'single' test mode")
            if any(size <= 0 for size in sizes):
                raise ValueError(f"(m, n, k) = {sizes}, all must be positive")
    except ValueError as arg_error:
        print(arg_error)
        sys.exit(1)
    return args


def main() -> None:
    args: argparse.Namespace = parse_args()

    torch.manual_seed(42)

    if args.mode == "full":
        sys.exit(pytest.main(["--quiet", "--exitfirst"]))

    run_config(args.m, args.n, args.k)


if __name__ == "__main__":
    main()
