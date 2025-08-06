import argparse

from common_cli import create_base_parser, shape_2d


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = create_base_parser("softmax", args)
    parser.add_argument(
        "shape",
        type=shape_2d,
        nargs="+",
        help="sofmax shape (pair of comma separated positive integers)",
    )
    return parser.parse_args(args)
