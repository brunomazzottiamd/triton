import argparse


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vector_add", description="run Triton 'vector_add' kernel"
    )
    parser.add_argument("n", type=int, nargs="+", help="vector size")
    parser.add_argument(
        "--runs", type=int, default=1, help="number of runs (default: 1)"
    )
    parser.add_argument(
        "--save-tensors", action="store_true", help="save tensors if this flag is set"
    )
    parser.add_argument("--verbose", action="store_true", help="enable verbose logging")
    parsed_args = parser.parse_args(args)
    if any(n <= 0 for n in parsed_args.n):
        parser.error("all values for vector size n must be positive integers")
    if parsed_args.runs <= 0:
        parser.error("number of runs must be a positive integer")
    return parsed_args
