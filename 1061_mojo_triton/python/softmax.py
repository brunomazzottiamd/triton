# Based on AITER softmax kernel:
# https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/softmax.py


import typing

import numpy as np
import torch
import triton
import triton.language as tl

from np_tensor import gen_tensor, save_tensor
from torch_interop import np_to_torch, torch_to_np
from softmax_cli import parse_args


@typing.no_type_check
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_idx = row_start

    # loop 1, find max and sum
    m = -float("inf")  # Initial value of max
    row_sum = 0.0
    row_start_ptr = input_ptr + row_idx * input_row_stride
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(
            input_ptrs, mask=mask, other=-float("inf"), cache_modifier=".cg"
        )  # load block
        m_p = tl.max(row_block, axis=0)  # find block max
        m_p = tl.maximum(m, m_p)  # Find new max across all blocks so far
        row_sum = row_sum * tl.exp(m - m_p)  # Adjust previous sum
        row_sum += tl.sum(
            tl.exp(row_block - m_p)
        )  # Add to exponentiated sum of this block
        m = m_p  # save max

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # Loop 2
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(
            input_ptrs, mask=mask, other=-float("inf"), cache_modifier=".cg"
        )  # load block
        # subtract, exponentiate and divide by sum
        softmax_output = tl.exp(row_block - m) / row_sum
        # store
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: np.ndarray) -> np.ndarray:
    t_x = np_to_torch(x)
    n_rows, n_cols = t_x.shape
    MAX_FUSED_SIZE = 65536 // t_x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    t_y = torch.empty_like(t_x)
    waves_per_eu = 2
    num_warps = 8
    num_stages = 2
    num_programs = n_rows
    grid = (num_programs,)
    softmax_kernel[grid](
        t_y,
        t_x,
        t_x.stride(0),
        t_y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    y = torch_to_np(t_y)
    return y


def run_softmax(
    shapes: list[tuple[int, int]], runs: int, save_tensors: bool, verbose: bool
) -> None:
    for shape in shapes:
        x = gen_tensor(shape)
        if save_tensors:
            if verbose:
                print(f"Saving softmax input for shape={shape}.")
            save_tensor(f"triton_softmax_x_{shape[0]:05d}_{shape[1]:05d}", x)
        if verbose:
            print(f"Running softmax for shape={shape} (1 / {runs}).")
        y = softmax(x)
        for i in range(2, runs + 1):
            if verbose:
                print(f"Running softmax for shape={shape} ({i} / {runs}).")
            y = softmax(x)
        if save_tensors:
            if verbose:
                print(f"Saving softmax output for shape={shape}.")
            save_tensor(f"triton_softmax_y_{shape[0]:05d}_{shape[1]:05d}", y)


def main() -> None:
    args = parse_args()
    run_softmax(args.shape, args.runs, args.save_tensors, args.verbose)


if __name__ == "__main__":
    main()
