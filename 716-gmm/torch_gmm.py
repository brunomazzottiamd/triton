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
from gmm_common import get_gmm_shape, get_gmm_output


# PyTorch reference GMM implementation.
# ------------------------------------------------------------------------------


def torch_gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    trans_out: bool = TRANS_OUT,
    existing_out: Tensor | None = None,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)

    M, _, N, G = get_gmm_shape(lhs, rhs, group_sizes)

    out = get_gmm_output(
        M,
        N,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        trans=trans_out,
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
