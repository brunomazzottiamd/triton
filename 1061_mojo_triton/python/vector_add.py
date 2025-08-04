import numpy as np
import torch
import triton
import triton.language as tl

from np_tensor import gen_tensor, save_tensor
from torch_interop import np_to_torch, torch_to_np
from vector_add_cli import parse_args


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = x + y
    tl.store(z_ptr + offs, z, mask=mask)


def vector_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    t_x = np_to_torch(x)
    t_y = np_to_torch(y)
    n = t_x.numel()
    t_z = torch.empty_like(t_x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](t_x, t_y, t_z, n, BLOCK_SIZE=1024)
    z = torch_to_np(t_z)
    return z


def run_vector_add(ns: list[int], runs: int, save_out: bool, verbose: bool) -> None:
    for n in ns:
        x = gen_tensor(n)
        y = gen_tensor(n, rng_seed=None)
        if save_out:
            if verbose:
                print(f"Saving vector add input for n={n}.")
            save_tensor(f"triton_vector_add_x_{n:09d}", x)
            save_tensor(f"triton_vector_add_y_{n:09d}", y)
        if verbose:
            print(f"Running vector add for n={n} (1 / {runs}).")
        z = vector_add(x, y)
        for i in range(2, runs + 1):
            if verbose:
                print(f"Running vector add for n={n} ({i} / {runs}).")
            z = vector_add(x, y)
        if save_out:
            if verbose:
                print(f"Saving vector add output for n={n}.")
            save_tensor(f"triton_vector_add_z_{n:09d}", z)


def main() -> None:
    args = parse_args()
    run_vector_add(args.n, args.runs, args.save_out, args.verbose)


if __name__ == "__main__":
    main()
