# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------


# Triton
import triton
import triton.language as tl


# General matrix multiplication tiling functions.
# ------------------------------------------------------------------------------


# Re-order tile ID for better L2 performance.
@triton.jit
def tile_grid(tile_in_mm, num_row_tiles, num_col_tiles, GROUP_SIZE: tl.constexpr = 1):
    if GROUP_SIZE == 1:
        row_tile = tile_in_mm // num_col_tiles
        col_tile = tile_in_mm % num_col_tiles
    else:
        num_tiles_in_group = GROUP_SIZE * num_col_tiles
        group_id = tile_in_mm // num_tiles_in_group
        first_row_tile = group_id * GROUP_SIZE
        group_row_size = min(num_row_tiles - first_row_tile, GROUP_SIZE)
        row_tile = first_row_tile + (tile_in_mm % group_row_size)
        col_tile = (tile_in_mm % num_tiles_in_group) // group_row_size

    tl.device_assert(row_tile >= 0, "row_tile < 0")
    tl.device_assert(row_tile < num_row_tiles, "row_tile >= num_row_tiles")

    tl.device_assert(col_tile >= 0, "col_tile < 0")
    tl.device_assert(col_tile < num_col_tiles, "col_tile >= num_col_tiles")

    return row_tile, col_tile
