# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# PyTorch
import torch

# pytest
import pytest

# Types module
from dtypes import dtype_from_str

# Common module
from gmm_common import gen_gmm_tensors

# GMM implementations
from torch_gmm import torch_gmm
from triton_gmm import triton_gmm

# Common test module
from test_common import (
    TEST_SHAPES,
    INPUT_DTYPES_STR,
    OUTPUT_DTYPES_STR,
    TRANS_RHS_STR,
    RNG_SEED_STR,
    trans_rhs_from_str,
    rng_seed_from_str,
    skip,
    num_group_sizes,
    use_triton_autotune,
    check_tensors,
)


# GMM unit tests.
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("M, K, N, G", TEST_SHAPES)
@pytest.mark.parametrize("in_dtype_str", INPUT_DTYPES_STR)
@pytest.mark.parametrize("out_dtype_str", OUTPUT_DTYPES_STR)
@pytest.mark.parametrize("trans_rhs_str", TRANS_RHS_STR)
@pytest.mark.parametrize("rng_seed_str", RNG_SEED_STR)
def test_gmm(
    quick_test: bool,
    M: int,
    K: int,
    N: int,
    G: int,
    in_dtype_str: str,
    out_dtype_str: str,
    trans_rhs_str: str,
    rng_seed_str: str,
):
    in_dtype = dtype_from_str(in_dtype_str)
    out_dtype = dtype_from_str(out_dtype_str)
    trans_rhs = trans_rhs_from_str(trans_rhs_str)
    rng_seed = rng_seed_from_str(rng_seed_str)

    skip(quick_test, in_dtype, out_dtype)

    lhs, rhs, multiple_group_sizes, out_torch = gen_gmm_tensors(
        M,
        K,
        N,
        G,
        num_group_sizes(quick_test),
        input_type=in_dtype,
        output_type=out_dtype,
        trans_rhs=trans_rhs,
        rng_seed=rng_seed,
        unif_group_sizes=True,  # 1st group_sizes in test is evenly distributed
    )
    out_triton = torch.empty_like(out_torch)

    autotune = use_triton_autotune(quick_test, M, K, N, G)

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
            autotune=autotune,
        )

        m = int(torch.sum(group_sizes).item())
        check_tensors(
            out_triton[:m],
            out_torch[:m],
            "Triton GMM doesn't match PyTorch reference GMM.",
        )
