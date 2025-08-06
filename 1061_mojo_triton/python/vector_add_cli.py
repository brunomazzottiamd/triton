import argparse

from common_cli import create_base_parser, positive_int


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = create_base_parser("vector_add", args)
    parser.add_argument("n", type=positive_int, nargs="+", help="vector size")
    return parser.parse_args(args)
