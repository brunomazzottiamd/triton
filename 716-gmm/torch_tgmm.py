# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# PyTorch
import torch
from torch import Tensor

# Types module
from dtypes import DTYPE

# Common module
from common import TRANS_OUT, check_input_device_dtype
from tgmm_common import get_tgmm_shape, get_tgmm_output


# PyTorch reference TGMM implementation.
# ------------------------------------------------------------------------------


def torch_tgmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    trans_out: bool = TRANS_OUT,
    existing_out: Tensor | None = None,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)

    M, K, N, G = get_tgmm_shape(lhs, rhs, group_sizes)

    out = get_tgmm_output(
        K,
        N,
        G,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        trans=trans_out,
        existing_out=existing_out,
    )

    last_col = 0

    for g in range(G):
        m = int(group_sizes[g].item())

        # Skip group if there are no columns assigned to the group.
        if m == 0:
            continue

        start_idx = last_col
        end_idx = last_col + m

        out[g] = (lhs[:, start_idx:end_idx] @ rhs[start_idx:end_idx, :]).to(
            preferred_element_type
        )

        last_col += m

    return out
