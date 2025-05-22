# -*- coding: utf-8 -*-


# PyTorch
import torch


# Supported data types, as strings.
SUPPORTED_DTYPES_STR: set[str] = {"fp16", "bf16"}


# Convert string data type to PyTorch data type.
def dtype_from_str(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.strip().lower()
    dtype_str = dtype_str[1:] if dtype_str[0] in {"i", "o"} else dtype_str
    assert (
        dtype_str in SUPPORTED_DTYPES_STR
    ), "String data type isn't in set of supported string data types."
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]


# Supported data types, as PyTorch types.
SUPPORTED_DTYPES: set[torch.dtype] = {
    dtype_from_str(dtype_str) for dtype_str in SUPPORTED_DTYPES_STR
}


# Convert PyTorch data type to string data type.
def str_from_dtype(dtype: torch.dtype) -> str:
    assert (
        dtype in SUPPORTED_DTYPES
    ), "PyTorch data type isn't in set of supported PyTorch data types."
    return {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]


# Default data type, as string.
DTYPE_STR: str = "bf16"
assert (
    DTYPE_STR in SUPPORTED_DTYPES_STR
), "Default string data type isn't in set of supported string data types."


# Default data type, as PyTorch type.
DTYPE: torch.dtype = dtype_from_str(DTYPE_STR)
