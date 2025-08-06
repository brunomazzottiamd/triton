import argparse
import functools


def to_int(value: str) -> int:
    # First try to convert to float to handle ".0" decimal notation.
    float_value = float(value)
    # Check if it's a whole number (no fractional part).
    int_value = int(float_value)
    if float_value != int_value:
        raise ValueError
    return int_value


def positive_int(value: str) -> int:
    error = argparse.ArgumentTypeError(f"'{value}' is not a positive integer")
    try:
        int_value = to_int(value)
    except ValueError:
        raise error
    if int_value <= 0:
        raise error
    return int_value


def shape(dims: int, value: str) -> tuple[int, ...]:
    assert dims > 1
    error = argparse.ArgumentTypeError(f"'{value}' is not a {dims}D shape")
    value_parts = value.split(",")
    if len(value_parts) != dims:
        raise error
    try:
        shape = tuple(to_int(value_part) for value_part in value_parts)
    except ValueError:
        raise error
    if any(shape_dim <= 0 for shape_dim in shape):
        raise error
    return shape


shape_2d = functools.partial(shape, 2)
shape_3d = functools.partial(shape, 3)


def create_base_parser(
    kernel: str, args: list[str] | None = None
) -> argparse.ArgumentParser:
    kernel_impl = "Triton" if args is None else "Mojo"
    parser = argparse.ArgumentParser(
        prog=kernel, description=f"run {kernel_impl} {kernel} kernel"
    )
    parser.add_argument(
        "--runs", type=positive_int, default=1, help="number of runs (default: 1)"
    )
    parser.add_argument(
        "--save-tensors", action="store_true", help="save tensors if this flag is set"
    )
    parser.add_argument("--verbose", action="store_true", help="enable verbose logging")
    return parser
