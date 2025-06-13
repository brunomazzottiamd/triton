# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------


# Python standard library
from functools import partial

# PyTorch
import torch
from torch import Tensor

# pytest
import pytest

# Types module
from dtypes import SUPPORTED_DTYPES_STR

# Common module
from common import REAL_SHAPES


# Common utilities used by GMM and TGMM tests.
# ------------------------------------------------------------------------------


# Shapes used only for test purposes.
# fmt: off
TEST_ONLY_SHAPES: list[tuple[int, int, int, int]] = [
    #  M,    K,    N,   G
    ( 10,    2,    3,   4),
    ( 32,   16,    8,   4),  # Test 1
    (512, 4096, 2048, 160),  # Test 2
]
# fmt: on


# Test shapes are test only + real ones.
TEST_SHAPES: list[tuple[int, int, int, int]] = TEST_ONLY_SHAPES + REAL_SHAPES


# Input and output types.
INPUT_DTYPES_STR: set[str] = {"i" + dtype_str for dtype_str in SUPPORTED_DTYPES_STR}
OUTPUT_DTYPES_STR: set[str] = {"o" + dtype_str for dtype_str in SUPPORTED_DTYPES_STR}


# Transpositions.

TRANS_LSH_STR: set[str] = {f"tlhs{b}" for b in {"F", "T"}}
TRANS_RHS_STR: set[str] = {f"trhs{b}" for b in {"F", "T"}}


def trans_from_str(trans_str: str, tensor_str: str) -> bool:
    assert tensor_str in {"lhs", "rhs"}, f"Invalid tensor string ({tensor_str})."
    return trans_str.replace(f"t{tensor_str}", "") == "T"


trans_lhs_from_str = partial(trans_from_str, tensor_str="lhs")
trans_rhs_from_str = partial(trans_from_str, tensor_str="rhs")


# RNG seed.

RNG_SEED_STR: set[str] = {f"rng{rng_seed}" for rng_seed in {77, 121}}


def rng_seed_from_str(rng_seed_str: str) -> int:
    rng_seed_int = int(rng_seed_str.replace("rng", ""))
    assert rng_seed_int >= 0, f"RNG seed must be non-negative (it's {rng_seed_int})."
    return rng_seed_int


# Quick test skip conditions.


def skip(
    quick_test: bool,
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    if not quick_test:
        return
    if (in_dtype == torch.float16 and out_dtype == torch.bfloat16) or (
        in_dtype == torch.bfloat16 and out_dtype == torch.float16
    ):
        pytest.skip("Skipping mixed fp16 / bf16 types to speed up test execution.")


# Generation of group sizes.


def num_group_sizes(quick_test: bool) -> int:
    # Reduce number of distinct group sizes in quick test.
    return 1 if quick_test else 5


# Usage of Triton autotuning feature.


def use_triton_autotune(quick_test: bool, M: int, K: int, N: int, G: int) -> bool:
    # Don't use autotune for test only shapes, don't use autotune in quick test.
    return not (((M, K, N, G) in TEST_ONLY_SHAPES) or quick_test)


# Tensor comparison.


def check_tensors(
    actual: Tensor,
    expected: Tensor,
    msg: str,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    if atol is None:
        atol = 5e-3
    else:
        assert atol > 0, f"Absolute tolerance must be positive (it's {atol})."
    if rtol is None:
        rtol = 1e-2
    else:
        assert rtol > 0, f"Relative tolerance must be positive (it's {rtol})."
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        msg=lambda torch_msg: f"{msg}\n\n{torch_msg}\n",
    )
