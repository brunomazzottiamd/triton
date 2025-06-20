#!/usr/bin/env python


# -*- coding: utf-8 -*-

# Disable [https://docs.astral.sh/ruff/rules/module-import-not-at-top-of-file/]
# ruff warning.
# ruff: noqa: E402


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import argparse
import logging

# NumPy
import numpy as np

# PyTorch
import torch

# JAX
import jax

# JAX GPU backend only.
jax.config.update("jax_platforms", "gpu")

import jax.numpy as jnp

# Disable annoying AITER warning about NUMA balancing.
logging.getLogger("aiter").disabled = True

# AITER MHA
from aiter.ops.triton.mha import flash_attn_func as aiter_mha

# Pallas  MHA
from jax.experimental.pallas.ops.gpu.attention import mha as pallas_mha

# AXLearn MHA
from axlearn.common.flash_attention.gpu_attention import flash_attention as axlearn_mha


# Global defaults.
# ------------------------------------------------------------------------------

KERNELS: set[str] = {"aiter", "pallas", "axlearn", "compare"}
KERNEL: str = "compare"
assert KERNEL in KERNELS

BATCH_SIZE: int = 2
SEQ_LEN: int = 8192
NUM_HEADS: int = 24
HEAD_SIZE: int = 128
SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE)
assert all(dim > 0 for dim in SHAPE)

RNG_SEED: int = 0


# Tensor generation and conversion.
# ------------------------------------------------------------------------------


def gen_tensor(
    shape: tuple[int, ...] = SHAPE, rng_seed: int | None = None
) -> np.ndarray:
    if rng_seed is not None:
        np.random.seed(rng_seed)
    return np.random.randn(*shape).astype(np.float16)


def gen_qkv(
    shape: tuple[int, ...] = SHAPE, rng_seed: int | None = RNG_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        gen_tensor(shape=shape, rng_seed=rng_seed),
        gen_tensor(shape=shape),
        gen_tensor(shape=shape),
    )


def np_to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).cuda()


def torch_to_np(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy()


def np_to_jax(x: np.ndarray) -> jax.Array:
    return jnp.array(x, device=jax.devices("gpu")[0])


def jax_to_np(x: jax.Array) -> np.ndarray:
    return np.array(x)


# Kernel execution.
# ------------------------------------------------------------------------------


def run_aiter_mha(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    return torch_to_np(
        aiter_mha(np_to_torch(q), np_to_torch(k), np_to_torch(v), causal=True)
    )


def run_pallas_mha(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    return jax_to_np(
        pallas_mha(
            np_to_jax(q), np_to_jax(k), np_to_jax(v), segment_ids=None, causal=True
        )
    )


def run_axlearn_mha(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    return jax_to_np(axlearn_mha(np_to_jax(q), np_to_jax(k), np_to_jax(v)))


# Result comparison.
# ------------------------------------------------------------------------------


def log_diff_percentage(diff: np.ndarray, epsilon: float) -> None:
    assert epsilon > 0
    num_within_threshold = np.sum(diff <= epsilon)
    assert num_within_threshold >= 0
    percentage_within_threshold = 100 * num_within_threshold / diff.size
    assert percentage_within_threshold > 0
    logging.info(
        "%6.2f%% of elements differ by at most %.2e",
        percentage_within_threshold,
        epsilon,
    )


def log_diff(aiter_o: np.ndarray, pallas_o: np.ndarray) -> None:
    diff = np.abs(aiter_o - pallas_o)
    logging.info("Minimum absolute difference: %.2f", np.min(diff))
    logging.info("Mean absolute difference: %.2f", np.mean(diff))
    logging.info("Maximum absolute difference: %.2f", np.max(diff))
    for exp in range(-3, 2):
        log_diff_percentage(diff, 10**exp)


# Command line interface parsing.
# ------------------------------------------------------------------------------


def positive_int(value: str) -> int:
    error = argparse.ArgumentTypeError(f"'{value}' is not a positive integer")
    try:
        # First try to convert to float to handle ".0" decimal notation.
        float_value = float(value)
        # Check if it's a whole number (no fractional part).
        if float_value != int(float_value):
            raise error
        int_value = int(float_value)
    except ValueError:
        raise error
    if int_value <= 0:
        raise error
    return int_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run MHA kernel", add_help=False)

    # Kernel to run:
    parser.add_argument(
        "-k",
        "--kernel",
        type=str.lower,
        choices=KERNELS,
        default=KERNEL,
        help=f"MHA kernel to run: Triton kernel from AITER, Pallas kernel from JAX, compare both kernels (default: {KERNEL})",
    )

    # Shape:
    parser.add_argument(
        "-b",
        "--batch-size",
        type=positive_int,
        default=BATCH_SIZE,
        help=f"batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "-s",
        "--seq-len",
        type=positive_int,
        default=SEQ_LEN,
        help=f"sequence length (default: {SEQ_LEN})",
    )
    parser.add_argument(
        "-h",
        "--num-heads",
        type=positive_int,
        default=NUM_HEADS,
        help=f"number of heads (default: {NUM_HEADS})",
    )
    parser.add_argument(
        "-d",
        "--head-size",
        type=positive_int,
        default=HEAD_SIZE,
        help=f"head size (default: {HEAD_SIZE})",
    )

    # Input generation:
    parser.add_argument(
        "-r",
        "--rng-seed",
        type=int,
        default=RNG_SEED,
        help=f"seed for random input generation (default: {RNG_SEED})",
    )

    # Other arguments:
    parser.add_argument("--verbose", action="store_true", help="enable verbose output")
    parser.add_argument("--help", action="help", help="show this help message and exit")

    args = parser.parse_args()
    args.shape = (args.batch_size, args.seq_len, args.num_heads, args.head_size)

    return args


# Main function: entry point.
# ------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s > %(message)s",
        level=logging.INFO if args.verbose else logging.ERROR,
        force=True,  # override previous logging configuration
    )

    logging.info("Generating tensors with shape %s...", args.shape)
    q, k, v = gen_qkv(shape=args.shape, rng_seed=args.rng_seed)

    if args.kernel in {"aiter", "compare"}:
        logging.info("Running AITER MHA...")
        aiter_o = run_aiter_mha(q, k, v)

    if args.kernel in {"pallas", "compare"}:
        logging.info("Running Pallas MHA...")
        pallas_o = run_pallas_mha(q, k, v)

    if args.kernel in {"axlearn", "compare"}:
        logging.info("Running AXLearn MHA...")
        axlearn_o = run_axlearn_mha(q, k, v)

    if args.kernel == "compare":
        logging.info("AITER output vs. Pallas output:")
        log_diff(aiter_o, pallas_o)
        logging.info("AITER output vs. AXLearn output:")
        log_diff(aiter_o, axlearn_o)
        logging.info("Pallas output vs. AXLearn output:")
        log_diff(pallas_o, axlearn_o)


if __name__ == "__main__":
    main()
