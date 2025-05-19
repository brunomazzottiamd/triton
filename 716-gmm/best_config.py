# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
from dataclasses import dataclass
import logging

# PyTorch
import torch

# Common module
from common import (
    SUPPORTED_DTYPES,
    DTYPE,
    TRANS_LHS,
    TRANS_RHS,
    TRANS_OUT,
    TILING,
    num_sms,
    is_power_of_2,
    get_tiling,
)


# Kernel configuration class.
# ------------------------------------------------------------------------------


@dataclass(frozen=True, eq=True)
class ConfigKey:
    M: int
    K: int
    N: int
    G: int
    input_type: torch.dtype = DTYPE
    output_type: torch.dtype = DTYPE
    trans_lhs: bool = TRANS_LHS
    trans_rhs: bool = TRANS_RHS
    trans_out: bool = TRANS_OUT

    def __post_init__(self):
        assert self.M > 0, f"Number of lhs rows M must be positive (M = {self.M})."
        assert (
            self.K > 0
        ), f"Number of lhs columns / rhs rows K must be positive (K = {self.K})."
        assert self.N > 0, f"Number of rhs columns N must be positive (N = {self.N})."
        assert self.G > 0, f"Number of groups G must be positive (G = {self.G})."
        assert (
            self.input_type in SUPPORTED_DTYPES
        ), "Input type must be supported by the kernel."
        assert (
            self.output_type in SUPPORTED_DTYPES
        ), "Output type must be supported by the kernel."


@dataclass(frozen=True, eq=True)
class Config:
    block_size_m: int = TILING[0]
    block_size_k: int = TILING[1]
    block_size_n: int = TILING[2]
    group_size_m: int = 1
    grid_dim: int = num_sms()
    num_warps: int = 4
    num_stages: int = 1

    def __post_init__(self):
        assert is_power_of_2(
            self.block_size_m
        ), f"M-dimension tile size must be a power of 2 (it's {self.block_size_m})."
        assert is_power_of_2(
            self.block_size_k
        ), f"K-dimension tile size must be a power of 2 (it's {self.block_size_k})."
        assert is_power_of_2(
            self.block_size_n
        ), f"N-dimension tile size must be a power of 2 (it's {self.block_size_n})."
        assert (
            self.group_size_m > 0
        ), f"Group size in M-dimension must be positive (it's {self.group_size_m})."
        assert (
            self.grid_dim > 0
        ), f"Grid dimension must be positive (it's {self.grid_dim})."
        assert (
            self.num_warps > 0
        ), f"Number of warps must be positive (it's {self.num_warps})."
        assert (
            self.num_stages >= 0
        ), f"Number of software pipeliner stages must be non-negative (it's {self.num_stages})."


# Database of best kernel configurations.
# ------------------------------------------------------------------------------


# Tuning database for gfx942.
# fmt: off
BEST_CONFIGS: dict[ConfigKey, Config] = {
    # bf16 bf16 TN
    ConfigKey(M=  49152, K= 1408, N= 2048, G=64): Config(block_size_m= 64, block_size_k=32, block_size_n=256, group_size_m=1, num_warps=8, num_stages=2),
    ConfigKey(M=3145728, K= 2048, N= 1408, G= 8): Config(block_size_m=128, block_size_k=32, block_size_n=256, group_size_m=2, num_warps=8, num_stages=1),
    ConfigKey(M= 393216, K= 2048, N= 1408, G=64): Config(block_size_m=128, block_size_k=32, block_size_n=256, group_size_m=4, num_warps=8, num_stages=1),
    ConfigKey(M=  32768, K= 6144, N=16384, G= 8): Config(block_size_m=128, block_size_k=32, block_size_n=256, group_size_m=2, num_warps=8, num_stages=1),
    ConfigKey(M=  32768, K=16384, N= 6144, G= 8): Config(block_size_m=128, block_size_k=32, block_size_n=256, group_size_m=2, num_warps=8, num_stages=1),
    # bf16 bf16 NN
    ConfigKey(M=  49152, K= 1408, N= 2048, G=64, trans_lhs=True): Config(block_size_m=128, block_size_k=32, block_size_n=128, group_size_m=2, num_warps=8, num_stages=2),
    ConfigKey(M=3145728, K= 2048, N= 1408, G= 8, trans_lhs=True): Config(block_size_m=256, block_size_k=32, block_size_n=128, group_size_m=4, num_warps=8, num_stages=1),
    ConfigKey(M= 393216, K= 2048, N= 1408, G=64, trans_lhs=True): Config(block_size_m=256, block_size_k=32, block_size_n=128, group_size_m=8, num_warps=4, num_stages=1),
    ConfigKey(M=  32768, K= 6144, N=16384, G= 8, trans_lhs=True): Config(block_size_m=256, block_size_k=32, block_size_n=128, group_size_m=8, num_warps=4, num_stages=1),
    ConfigKey(M=  32768, K=16384, N= 6144, G= 8, trans_lhs=True): Config(block_size_m=256, block_size_k=32, block_size_n=128, group_size_m=2, num_warps=4, num_stages=1),
    # bf16 bf16 NT
    ConfigKey(M=  49152, K= 1408, N= 2048, G=64, trans_lhs=True, trans_rhs=False): Config(block_size_m=256, block_size_k=32, block_size_n=128, group_size_m=8, num_warps=4, num_stages=1),
    ConfigKey(M=3145728, K= 2048, N= 1408, G= 8, trans_lhs=True, trans_rhs=False): Config(block_size_m=256, block_size_k=32, block_size_n=128, group_size_m=8, num_warps=8, num_stages=1),
    ConfigKey(M= 393216, K= 2048, N= 1408, G=64, trans_lhs=True, trans_rhs=False): Config(block_size_m=256, block_size_k=32, block_size_n=128, group_size_m=8, num_warps=4, num_stages=1),
    ConfigKey(M=  32768, K= 6144, N=16384, G= 8, trans_lhs=True, trans_rhs=False): Config(block_size_m=256, block_size_k=32, block_size_n=128, group_size_m=2, num_warps=4, num_stages=1),
    ConfigKey(M=  32768, K=16384, N= 6144, G= 8, trans_lhs=True, trans_rhs=False): Config(block_size_m=256, block_size_k=32, block_size_n=128, group_size_m=8, num_warps=8, num_stages=2),
}
# fmt: on


# Selection of best kernel configuration.
# ------------------------------------------------------------------------------


def pick_best_config(
    M: int,
    K: int,
    N: int,
    G: int,
    group_sizes: torch.Tensor | None,
    input_type: torch.dtype = DTYPE,
    output_type: torch.dtype = DTYPE,
    trans_lhs: bool = TRANS_LHS,
    trans_rhs: bool = TRANS_RHS,
    trans_out: bool = TRANS_OUT,
) -> Config:
    config_key = ConfigKey(
        M, K, N, G, input_type, output_type, trans_lhs, trans_rhs, trans_out
    )
    logging.debug("Querying best config for %s.", config_key)
    try:
        best_config = BEST_CONFIGS[config_key]
    except KeyError:
        logging.debug(
            "Could not find best config for %s, picking default one + block size heuristics.",
            config_key,
        )
        block_size_m, block_size_k, block_size_n = get_tiling(
            M, K, N, TILING, group_sizes=group_sizes
        )
        best_config = Config(
            block_size_m=block_size_m,
            block_size_k=block_size_k,
            block_size_n=block_size_n,
        )
    logging.debug("Best config for %s is %s.", config_key, best_config)
    return best_config
