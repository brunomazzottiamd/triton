# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------


# PyTorch
import torch

# pytest
import pytest

# Common module
from common import (
    SUPPORTED_DTYPES_STR,
    TILING,
    dtype_from_str,
    gen_input,
    gen_multiple_group_sizes,
    gen_output,
)

# GMM implementations
from torch_gmm import torch_gmm
from triton_gmm import triton_gmm


# Unit tests.
# ------------------------------------------------------------------------------


# GMM shapes used only for test purposes,
# fmt: off
TEST_ONLY_SHAPES: list[tuple[int, int, int, int]] = [
    #  M,    K,    N,   G
    ( 10,    2,    3,   4),
    ( 32,   16,    8,   4),  # Test 1
    (512, 4096, 2048, 160),  # Test 2
]
# fmt: on


# Real GMM shapes, used by real models.
# fmt: off
REAL_SHAPES: list[tuple[int, int, int, int]] = [
    #      M,     K,     N,   G
    (  49152,  1408,  2048,  64),  # deepseekv2-16B
    (3145728,  2048,  1408,   8),  # deepseekv2-16B (IT'S BIG! I was getting core dump with this shape! lhs => 12 GB, out => 8.25 GB)
    ( 393216,  2048,  1408,  64),  # deepseekv2-16B
    (  32768,  6144, 16384,   8),  # Mixtral 8x22B proxy model
    (  32768, 16384,  6144,   8),  # Mixtral 8x22B proxy model
]
# fmt: on


QUICK_TEST: bool = False


@pytest.mark.parametrize("M, K, N, G", TEST_ONLY_SHAPES + REAL_SHAPES)
@pytest.mark.parametrize(
    "in_dtype_str", {"i" + dtype_str for dtype_str in SUPPORTED_DTYPES_STR}
)
@pytest.mark.parametrize(
    "out_dtype_str", {"o" + dtype_str for dtype_str in SUPPORTED_DTYPES_STR}
)
@pytest.mark.parametrize("trans_lhs_str", {f"tlhs{b}" for b in {"F", "T"}})
@pytest.mark.parametrize("trans_rhs_str", {f"trhs{b}" for b in {"F", "T"}})
@pytest.mark.parametrize("trans_out_str", {f"tout{b}" for b in {"F", "T"}})
@pytest.mark.parametrize("rng_seed_str", {f"rng{rng_seed}" for rng_seed in {77, 121}})
def test_gmm(
    M: int,
    K: int,
    N: int,
    G: int,
    in_dtype_str: str,
    out_dtype_str: str,
    trans_lhs_str: str,
    trans_rhs_str: str,
    trans_out_str: str,
    rng_seed_str: str,
):
    in_dtype = dtype_from_str(in_dtype_str)
    out_dtype = dtype_from_str(out_dtype_str)
    trans_lhs = trans_lhs_str.replace("tlhs", "") == "T"
    trans_rhs = trans_rhs_str.replace("trhs", "") == "T"
    trans_out = trans_out_str.replace("tout", "") == "T"
    rng_seed = int(rng_seed_str.replace("rng", ""))

    # Skip conditions:
    if QUICK_TEST and (
        (in_dtype == torch.float16 and out_dtype == torch.bfloat16)
        or (in_dtype == torch.bfloat16 and out_dtype == torch.float16)
    ):
        pytest.skip("Skipping mixed fp16 / bf16 types to speed up test execution.")
        # Important notice: mixed fp16 / bf16 types work correctly!

    lhs, rhs, group_sizes_0 = gen_input(
        M,
        K,
        N,
        G,
        preferred_element_type=in_dtype,
        trans_lhs=trans_lhs,
        trans_rhs=trans_rhs,
        rng_seed=rng_seed,
    )
    # Reduce number of distinct group sizes in quick test.
    num_group_sizes = 1 if QUICK_TEST else 5
    multiple_group_sizes = gen_multiple_group_sizes(
        num_group_sizes, M, G, rng_seed=None, group_sizes_0=group_sizes_0
    )

    out_torch = gen_output(M, N, preferred_element_type=out_dtype, trans=trans_out)
    out_triton = gen_output(M, N, preferred_element_type=out_dtype, trans=trans_out)

    # Don't use autotune for test only shapes, don't use autotune in quick test.
    tiling = TILING if (M, K, N, G) in TEST_ONLY_SHAPES or QUICK_TEST else None

    for group_sizes in multiple_group_sizes:
        torch_gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out_torch,
        )

        triton_gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out_triton,
            tiling=tiling,
        )

        torch.testing.assert_close(out_torch, out_triton, atol=5e-3, rtol=1e-2)
