import argparse

import numpy as np

import torch

import triton
import triton.language as tl

from common import gen_tensor, save_tensor, np_to_torch, torch_to_np


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    out = x + y
    tl.store(out_ptr + offs, out, mask=mask)


def torch_vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


def np_vector_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return torch_to_np(torch_vector_add(np_to_torch(x), np_to_torch(y)))


def run_vector_add(ns: list[int], runs: int, save_out: bool) -> None:
    for n in ns:
        x = gen_tensor(n)
        y = gen_tensor(n, rng_seed=None)
        for _ in range(0, runs):
            out = np_vector_add(x, y)
        if save_out:
            save_tensor(f"triton_vector_add_{n:08d}", out)


def parse_args():
    parser = argparse.ArgumentParser(description="run Triton 'vector_add' kernel")
    parser.add_argument("n", type=int, nargs="+", help="vector size")
    parser.add_argument(
        "--runs", type=int, default=1, help="number of runs (default: 1)"
    )
    parser.add_argument(
        "--save-out", action="store_true", help="save output if this flag is set"
    )
    args = parser.parse_args()
    if any(n <= 0 for n in args.n):
        parser.error("all values for vector size n must be positive integers")
    if args.runs <= 0:
        parser.error("number of runs must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    run_vector_add(args.n, args.runs, args.save_out)


if __name__ == "__main__":
    main()
