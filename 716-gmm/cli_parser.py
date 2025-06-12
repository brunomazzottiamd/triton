# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import argparse
import itertools

# Types module
from dtypes import (
    SUPPORTED_DTYPES_STR,
    DTYPE_STR,
)

# Common module
from common import (
    TRANS_LHS,
    TRANS_RHS,
    TRANS_OUT,
    RNG_SEED,
    NUM_GROUP_SIZES,
)


# Command line interface parsing.
# ------------------------------------------------------------------------------


def positive_int(value: str) -> int:
    error = argparse.ArgumentTypeError(f"'{value}' is not a positive integer")
    try:
        # First try to convert to float to handle ".0" decimal notation.
        float_value = float(value)
        # Check if it's a whole number (no fractional part).
        if float_value != int(float_value):
            raise error
        int_value = int(float_value)
    except ValueError:
        raise error
    if int_value <= 0:
        raise error
    return int_value


def add_trans_arg(
    parser: argparse.ArgumentParser, arg: str, default_trans: bool
) -> None:
    if default_trans:
        parser.add_argument(
            f"--no-trans-{arg}",
            action="store_false",
            dest=f"trans_{arg}",
            help=f"don't transpose {arg}, i.e. row-major {arg}",
        )
    else:
        parser.add_argument(
            f"--trans-{arg}",
            action="store_true",
            dest=f"trans_{arg}",
            help=f"transpose {arg}, i.e. column-major {arg}",
        )


def layout_choices() -> set[str]:
    row_col_layout_chars = {"r", "c"}
    return {
        f"{lhs}{rhs}{out}"
        for lhs, rhs, out in itertools.product(
            row_col_layout_chars, row_col_layout_chars, row_col_layout_chars
        )
    } | {"nn", "tn", "nt"}


def trans_from_layout(layout: str) -> tuple[bool, ...]:
    assert layout in layout_choices(), "Invalid matrix multiplication layout."
    try:
        layout = {"nn": "rrr", "tn": "crr", "nt": "rcr"}[layout]
    except KeyError:
        pass
    assert (
        len(layout) == 3
    ), "Row / column layout string must have exactly 3 characters."
    assert all(
        layout_char in {"r", "c"} for layout_char in layout
    ), "All row / column layout characters must be 'r' or 'c'."
    return tuple(layout_char == "c" for layout_char in layout)


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    shape_args = [args.M, args.K, args.N, args.G]
    all_none = all(arg is None for arg in shape_args)
    all_provided = all(arg is not None for arg in shape_args)

    if args.bench:
        if not all_none and not all_provided:
            raise argparse.ArgumentError(
                None,
                "when --bench is used, M, K, N, and G must be either all provided or all absent",
            )
    else:
        if not all_provided:
            raise argparse.ArgumentError(
                None, "M, K, N, and G are mandatory when --bench isn't used"
            )

    if args.unif_group_sizes and args.num_group_sizes != 1:
        raise argparse.ArgumentError(
            None,
            "number of distinct group sizes must be 1 when --unif-group-sizes is used",
        )

    if args.layout is not None:
        # Validate layout and transposition combinations.
        has_trans_args = (
            args.trans_lhs != TRANS_LHS
            or args.trans_rhs != TRANS_RHS
            or args.trans_out != TRANS_OUT
        )
        if has_trans_args:
            raise argparse.ArgumentError(
                None, "transposition arguments aren't supported when --layout is used"
            )
        args.trans_lhs, args.trans_rhs, args.trans_out = trans_from_layout(args.layout)

    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run GMM Triton kernel")

    # Shape
    parser.add_argument("M", type=positive_int, nargs="?", help="number of rows")
    parser.add_argument("K", type=positive_int, nargs="?", help="shared dimension")
    parser.add_argument("N", type=positive_int, nargs="?", help="number of columns")
    parser.add_argument("G", type=positive_int, nargs="?", help="number of groups")

    # GMM type
    parser.add_argument(
        "--gmm-type",
        choices={"gmm", "ptgmm", "tgmm"},
        default="gmm",
        help="GMM variant to run: GMM, persistent TGMM, non-persistent TGMM",
    )

    # Data type
    parser.add_argument(
        "--input-type",
        choices=SUPPORTED_DTYPES_STR,
        default=DTYPE_STR,
        help=f"input data type (default: {DTYPE_STR})",
    )
    parser.add_argument(
        "--output-type",
        choices=SUPPORTED_DTYPES_STR,
        default=DTYPE_STR,
        help=f"output data type (default: {DTYPE_STR})",
    )

    # Transpose and layout
    add_trans_arg(parser, "lhs", TRANS_LHS)
    add_trans_arg(parser, "rhs", TRANS_RHS)
    add_trans_arg(parser, "out", TRANS_OUT)
    parser.add_argument(
        "--layout",
        type=str.lower,
        choices=layout_choices(),
        help="matrix multiplication memory layout, should not be used together with transposition arguments",
    )

    # Input generation
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=RNG_SEED,
        help=f"seed for random input generation (default: {RNG_SEED})",
    )
    parser.add_argument(
        "--num-group-sizes",
        type=positive_int,
        default=NUM_GROUP_SIZES,
        help=f"number of distinct random group sizes to use (default: {NUM_GROUP_SIZES})",
    )
    parser.add_argument(
        "--unif-group-sizes",
        action="store_true",
        help="evenly distributes tokens among all groups",
    )

    # Other arguments
    parser.add_argument(
        "--bench", action="store_true", help="benchmark kernel instead of running it"
    )
    parser.add_argument("--verbose", action="store_true", help="enable verbose output")

    try:
        return validate_args(parser.parse_args())
    except argparse.ArgumentError as arg_error:
        import sys

        parser.print_usage()
        print(f"{sys.argv[0]}: error: {arg_error}")
        sys.exit(1)


if __name__ == "__main__":
    print(parse_args())
