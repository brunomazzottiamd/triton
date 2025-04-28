# -*- coding: utf-8 -*-


# GMM problem description:
# * Input tensors:
#   * lhs is (M, K) bf16
#   * rhs is (G, K, N) bf16
#   * group_sizes is (G,) int32
# * Output tensors:
#   * out is (M, N)


import random

import torch
from torch import Tensor

import triton
import triton.language as tl

import pytest


DEVICE: str = "cuda"


def random_group_sizes(M: int, G: int, rng_seed: int | None = None) -> list[int]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."
    assert G <= M, f"Cannot split M into more than M groups (M = {M}, G = {G})."

    if rng_seed is not None:
        random.seed(rng_seed)

    # Generate G - 1 sorted cut points between 1 and M - 1.
    cuts = sorted(random.sample(range(1, M), G - 1))
    # Add 0 at the beginning and M at the end, then take differences.
    group_sizes = [b - a for a, b in zip([0] + cuts, cuts + [M])]

    assert len(group_sizes) == G
    assert sum(group_sizes) == M

    return group_sizes


def gen_input(
    M: int, K: int, N: int, G: int, rng_seed: int | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    lhs = torch.randn((M, K), dtype=torch.float32, device=DEVICE)
    lhs = lhs.to(torch.bfloat16)

    rhs = torch.randn((G, K, N), dtype=torch.float32, device=DEVICE)
    rhs = rhs.to(torch.bfloat16)

    group_sizes = random_group_sizes(M, G, rng_seed=rng_seed)
    group_sizes = torch.tensor(group_sizes, dtype=torch.int32, device=DEVICE)

    return lhs, rhs, group_sizes


def gen_output(M: int, N: int) -> Tensor:
    assert M > 0, f"Number of out rows M must be positive (M = {M})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."

    out = torch.empty((M, N), dtype=torch.bfloat16, device=DEVICE)

    return out


def shapes_from_input(
    lhs: Tensor, rhs: Tensor, group_sizes: Tensor
) -> tuple[int, int, int, int]:
    assert lhs.dim() == 2, f"lhs must have 2 dimensions (lhs.dim() = {lhs.dim()})."
    assert rhs.dim() == 3, f"rhs must have 3 dimensions (rhs.dim() = {rhs.dim()})."
    assert (
        group_sizes.dim() == 1
    ), f"group_sizes must have 1 dimension (group_sizes.dim() = {group_sizes.dim()})."

    M, lhs_k = lhs.shape
    rhs_g, rhs_k, N = rhs.shape
    group_sizes_g = group_sizes.shape[0]

    assert (
        lhs_k == rhs_k
    ), f"K dimension of lhs and rhs don't match ({lhs_k} != {rhs_k})."
    K = lhs_k
    assert (
        rhs_g == group_sizes_g
    ), f"G dimension of rhs and group_sizes don't match ({rhs_g} != {group_sizes_g})."
    G = rhs_g

    return M, K, N, G


def torch_gmm(
    lhs: Tensor, rhs: Tensor, group_sizes: Tensor, out: Tensor | None = None
) -> Tensor:
    M, K, N, G = shapes_from_input(lhs, rhs, group_sizes)

    if out is None:
        out = gen_output(M, N)

    return out


@pytest.mark.parametrize(
    "M, K, N, G",
    [
        (32, 16, 8, 4),  # Test 1
        (512, 4096, 2048, 160),  # Test 2
        (49152, 1408, 2048, 64),  # deepseekv2-16B
        (3145728, 2048, 1408, 8),  # deepseekv2-16B
        (393216, 2048, 1408, 64),  # deepseekv2-16B
        (32768, 6144, 16384, 8),  # Mixtral 8x22B proxy model
        (32768, 16384, 6144, 8),  # Mixtral 8x22B proxy model
    ],
)
def test_gmm(M: int, K: int, N: int, G: int):
    lhs, rhs, group_sizes = gen_input(M, K, N, G, rng_seed=0)
    out_torch = torch_gmm(lhs, rhs, group_sizes)
