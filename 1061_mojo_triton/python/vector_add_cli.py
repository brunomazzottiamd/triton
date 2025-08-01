import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="run Triton 'vector_add' kernel")
    parser.add_argument("n", type=int, nargs="+", help="vector size")
    parser.add_argument(
        "--runs", type=int, default=1, help="number of runs (default: 1)"
    )
    parser.add_argument(
        "--save-out", action="store_true", help="save output if this flag is set"
    )
    args = parser.parse_args()
    if any(n <= 0 for n in args.n):
        parser.error("all values for vector size n must be positive integers")
    if args.runs <= 0:
        parser.error("number of runs must be a positive integer")
    return args
