# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import itertools

# Triton
import triton

# Common module
from common import num_sms


# Common tuning.
# ------------------------------------------------------------------------------


# Generate lots of configs with Cartesian product approach.
def full_tuning_space(add_grid_dim: bool = True) -> list[triton.Config]:
    block_sizes = [32, 64, 128, 256]
    # GMM specific:
    # |_ Consider restricting block_size_m_range to [64, 128, 256].
    # |_ Consider restricting block_size_k_range to [32].
    # |_ Consider restricting block_size_n_range to [128, 256].
    group_size_range = [1, 2, 4, 8]
    grid_dim_range = [sms_multiplier * num_sms() for sms_multiplier in range(1, 5)]
    num_warps_range = [2, 4, 8]
    # GMM specific:
    # |_ Consider restricting num_warps_range to [4, 8].
    num_stages_range = [1, 2]

    # Other parameters to tune:
    # waves_per_eu_range = [0, 2, 4, 8]
    # matrix_instr_nonkdim_range = [16, 32]
    # kpack_range = [1, 2]

    # WARNING: 6144 configurations per shape with grid_dim!
    #          1536 configurations per shape without grid_dim!

    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_K": block_size_k,
                "BLOCK_SIZE_N": block_size_n,
                "GROUP_SIZE": group_size,
                # "waves_per_eu": waves_per_eu,
                # "kpack": kpack,
                # "matrix_instr_nonkdim": matrix_instr_nonkdim,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_size_m, block_size_k, block_size_n, group_size, num_warps, num_stages in itertools.product(
            block_sizes,
            block_sizes,
            block_sizes,
            group_size_range,
            num_warps_range,
            num_stages_range,
        )
    ]

    if add_grid_dim:
        configs = [
            triton.Config(
                config.kwargs | {"GRID_DIM": grid_dim},
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
            for config, grid_dim in itertools.product(configs, grid_dim_range)
        ]

    return configs
