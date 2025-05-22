# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# PyTorch
import torch
from torch import Tensor

# Types module
from dtypes import DTYPE

# Common module
from common import check_input_device_dtype, get_shape_from_input, get_output


# PyTorch reference GMM implementation.
# ------------------------------------------------------------------------------


def torch_gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)
    M, _, N, G = get_shape_from_input(lhs, rhs, group_sizes)

    out = get_output(
        M,
        N,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        existing_out=existing_out,
    )

    last_row = 0

    for g in range(G):
        m = int(group_sizes[g].item())

        # Skip group if there are no tokens assigned to the expert.
        if m == 0:
            continue

        start_idx = last_row
        end_idx = last_row + m

        out[start_idx:end_idx, :] = (lhs[start_idx:end_idx, :] @ rhs[g]).to(
            preferred_element_type
        )

        last_row += m

    return out
