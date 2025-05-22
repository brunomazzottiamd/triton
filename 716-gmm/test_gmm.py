# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------


# PyTorch
import torch

# pytest
import pytest

# Types module
from dtypes import SUPPORTED_DTYPES_STR, dtype_from_str

# Common module
from common import REAL_SHAPES
from gmm_common import gen_gmm_input, gen_gmm_output

# Group sizes module
from group_sizes import gen_multiple_group_sizes

# GMM implementations
from torch_gmm import torch_gmm
from triton_gmm import triton_gmm


# Unit tests.
# ------------------------------------------------------------------------------


# Shapes used only for test purposes,
# fmt: off
TEST_ONLY_SHAPES: list[tuple[int, int, int, int]] = [
    #  M,    K,    N,   G
    ( 10,    2,    3,   4),
    ( 32,   16,    8,   4),  # Test 1
    (512, 4096, 2048, 160),  # Test 2
]
# fmt: on


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
    trans_lhs = trans_lhs_str.replace("tlhs", "") == "T"
    trans_rhs = trans_rhs_str.replace("trhs", "") == "T"
    trans_out = trans_out_str.replace("tout", "") == "T"
    rng_seed = int(rng_seed_str.replace("rng", ""))

    # Quick test skip conditions:
    if quick_test:
        if (in_dtype == torch.float16 and out_dtype == torch.bfloat16) or (
            in_dtype == torch.bfloat16 and out_dtype == torch.float16
        ):
            pytest.skip("Skipping mixed fp16 / bf16 types to speed up test execution.")
        if trans_out:
            pytest.skip("Skipping transposed output matrix to speed up test execution.")
        if (trans_lhs, trans_rhs) not in {(False, True), (True, False), (True, True)}:
            pytest.skip("Skipping non-{TN,NT,NN} layouts speed up test execution.")

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
    # Reduce number of distinct group sizes in quick test.
    num_group_sizes = 1 if quick_test else 5
    multiple_group_sizes = gen_multiple_group_sizes(
        num_group_sizes, M, G, rng_seed=None, group_sizes_0=group_sizes_0
    )

    out_torch = gen_gmm_output(M, N, preferred_element_type=out_dtype, trans=trans_out)
    out_triton = gen_gmm_output(M, N, preferred_element_type=out_dtype, trans=trans_out)

    # Don't use autotune for test only shapes, don't use autotune in quick test.
    autotune = not (((M, K, N, G) in TEST_ONLY_SHAPES) or quick_test)

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

        torch.testing.assert_close(out_torch, out_triton, atol=5e-3, rtol=1e-2)
