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
from dtypes import SUPPORTED_DTYPES_STR, dtype_from_str

# Common module
from common import REAL_SHAPES
from gmm_common import gen_gmm_input, gen_gmm_output
from tgmm_common import gen_tgmm_input, gen_tgmm_output

# Group sizes module
from group_sizes import gen_multiple_group_sizes

# GMM implementations
from torch_gmm import torch_gmm
from triton_gmm import triton_gmm

# TGMM implementations
from torch_tgmm import torch_tgmm
from triton_tgmm import triton_tgmm


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
TRANS_OUT_STR: set[str] = {f"tout{b}" for b in {"F", "T"}}


def trans_from_str(trans_str: str, tensor_str: str) -> bool:
    assert tensor_str in {"lhs", "rhs", "out"}, f"Invalid tensor string ({tensor_str})."
    return trans_str.replace(f"t{tensor_str}", "") == "T"


trans_lhs_from_str = partial(trans_from_str, tensor_str="lhs")
trans_rhs_from_str = partial(trans_from_str, tensor_str="rhs")
trans_out_from_str = partial(trans_from_str, tensor_str="out")


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
    trans_lhs: bool,
    trans_rhs: bool,
    trans_out: bool,
) -> None:
    if not quick_test:
        return
    if (in_dtype == torch.float16 and out_dtype == torch.bfloat16) or (
        in_dtype == torch.bfloat16 and out_dtype == torch.float16
    ):
        pytest.skip("Skipping mixed fp16 / bf16 types to speed up test execution.")
    if trans_out:
        pytest.skip("Skipping transposed output matrix to speed up test execution.")
    if (trans_lhs, trans_rhs) not in {(False, True), (True, False), (True, True)}:
        pytest.skip("Skipping non-{TN,NT,NN} layouts speed up test execution.")


# Generation of group sizes.


def gen_group_sizes(
    quick_test: bool, M: int, G: int, group_sizes_0: Tensor
) -> list[Tensor]:
    # Reduce number of distinct group sizes in quick test.
    num_group_sizes = 1 if quick_test else 5
    multiple_group_sizes = gen_multiple_group_sizes(
        num_group_sizes, M, G, rng_seed=None, group_sizes_0=group_sizes_0
    )
    return multiple_group_sizes


# Usage of Triton autotuning feature.


def use_triton_autotune(quick_test: bool, M: int, K: int, N: int, G: int) -> bool:
    # Don't use autotune for test only shapes, don't use autotune in quick test.
    return not (((M, K, N, G) in TEST_ONLY_SHAPES) or quick_test)


# Tensor comparison.


def check_tensors(actual: Tensor, expected: Tensor, msg: str) -> None:
    torch.testing.assert_close(
        actual,
        expected,
        atol=5e-3,
        rtol=1e-2,
        msg=lambda torch_msg: f"{msg}\n\n{torch_msg}\n",
    )


# GMM unit tests.
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("M, K, N, G", TEST_SHAPES)
@pytest.mark.parametrize("in_dtype_str", INPUT_DTYPES_STR)
@pytest.mark.parametrize("out_dtype_str", OUTPUT_DTYPES_STR)
@pytest.mark.parametrize("trans_lhs_str", TRANS_LSH_STR)
@pytest.mark.parametrize("trans_rhs_str", TRANS_RHS_STR)
@pytest.mark.parametrize("trans_out_str", TRANS_OUT_STR)
@pytest.mark.parametrize("rng_seed_str", RNG_SEED_STR)
def test_gmm(
    quick_test: bool,
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
    trans_lhs = trans_lhs_from_str(trans_lhs_str)
    trans_rhs = trans_rhs_from_str(trans_rhs_str)
    trans_out = trans_out_from_str(trans_out_str)
    rng_seed = rng_seed_from_str(rng_seed_str)

    skip(quick_test, in_dtype, out_dtype, trans_lhs, trans_rhs, trans_out)

    lhs, rhs, group_sizes_0 = gen_gmm_input(
        M,
        K,
        N,
        G,
        preferred_element_type=in_dtype,
        trans_lhs=trans_lhs,
        trans_rhs=trans_rhs,
        rng_seed=rng_seed,
        unif_group_sizes=True,  # 1st group_sizes in test is evenly distributed
    )
    multiple_group_sizes = gen_group_sizes(quick_test, M, G, group_sizes_0)

    out_torch = gen_gmm_output(M, N, preferred_element_type=out_dtype, trans=trans_out)
    out_triton = gen_gmm_output(M, N, preferred_element_type=out_dtype, trans=trans_out)

    autotune = use_triton_autotune(quick_test, M, K, N, G)

    for group_sizes in multiple_group_sizes:
        torch_gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            trans_out=trans_out,
            existing_out=out_torch,
        )

        triton_gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            trans_out=trans_out,
            existing_out=out_triton,
            autotune=autotune,
        )

        check_tensors(
            out_triton, out_torch, "Triton GMM doesn't match PyTorch reference GMM."
        )


# TGMM unit tests.
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("M, K, N, G", TEST_SHAPES)
@pytest.mark.parametrize("in_dtype_str", INPUT_DTYPES_STR)
@pytest.mark.parametrize("out_dtype_str", OUTPUT_DTYPES_STR)
@pytest.mark.parametrize("trans_lhs_str", TRANS_LSH_STR)
@pytest.mark.parametrize("trans_rhs_str", TRANS_RHS_STR)
@pytest.mark.parametrize("trans_out_str", TRANS_OUT_STR)
@pytest.mark.parametrize("rng_seed_str", RNG_SEED_STR)
def test_tgmm(
    quick_test: bool,
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
    trans_lhs = trans_lhs_from_str(trans_lhs_str)
    trans_rhs = trans_rhs_from_str(trans_rhs_str)
    trans_out = trans_out_from_str(trans_out_str)
    rng_seed = rng_seed_from_str(rng_seed_str)

    skip(quick_test, in_dtype, out_dtype, trans_lhs, trans_rhs, trans_out)

    lhs, rhs, group_sizes_0 = gen_tgmm_input(
        M,
        K,
        N,
        G,
        preferred_element_type=in_dtype,
        trans_lhs=trans_lhs,
        trans_rhs=trans_rhs,
        rng_seed=rng_seed,
        unif_group_sizes=True,  # 1st group_sizes in test is evenly distributed
    )
    multiple_group_sizes = gen_group_sizes(quick_test, M, G, group_sizes_0)

    out_torch = gen_tgmm_output(
        K, N, G, preferred_element_type=out_dtype, trans=trans_out
    )
    out_triton = gen_tgmm_output(
        K, N, G, preferred_element_type=out_dtype, trans=trans_out
    )

    autotune = use_triton_autotune(quick_test, M, K, N, G)

    for group_sizes in multiple_group_sizes:
        torch_tgmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            trans_out=trans_out,
            existing_out=out_torch,
        )

        triton_tgmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            trans_out=trans_out,
            existing_out=out_triton,
            autotune=autotune,
        )

        check_tensors(
            out_triton, out_torch, "Triton TGMM doesn't match PyTorch reference TGMM."
        )
