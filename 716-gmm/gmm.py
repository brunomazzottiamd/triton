#!/usr/bin/env python


# -*- coding: utf-8 -*-


# GMM problem description:
# * Input tensors:
#   * lhs is (M, K) bf16
#   * rhs is (G, K, N) bf16
#   * group_sizes is (G,) int32
# * Output tensors:
#   * out is (M, N) bf16


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import argparse
import logging

# PyTorch
import torch

# Triton
import triton

# Common module
from common import (
    SUPPORTED_DTYPES_STR,
    DTYPE_STR,
    DTYPE,
    TRANS_LHS,
    TRANS_RHS,
    TRANS_OUT,
    RNG_SEED,
    NUM_GROUP_SIZES,
    TILING,
    REAL_SHAPES,
    dtype_from_str,
    str_from_dtype,
    gen_input,
    gen_multiple_group_sizes,
    gen_output,
)

# Triton GMM implementations
from triton_gmm import triton_gmm, autotune_configs, triton_autotuned_gmm_kernel


# Benchmark.
# ------------------------------------------------------------------------------


def benchmark_triton_gmm(
    bench_shape: tuple[int, int, int, int] | None = None,
    in_dtype: torch.dtype = DTYPE,
    out_dtype: torch.dtype = DTYPE,
    trans_lhs: bool = TRANS_LHS,
    trans_rhs: bool = TRANS_RHS,
    trans_out: bool = TRANS_OUT,
    rng_seed: int = RNG_SEED,
    num_group_sizes: int = NUM_GROUP_SIZES,
) -> None:
    in_dtype_str = str_from_dtype(in_dtype)
    out_dtype_str = str_from_dtype(out_dtype)
    dtypes_desc = f"i{in_dtype_str}_o{out_dtype_str}"
    triton_provider = f"triton_{dtypes_desc}"

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "K", "N", "G"],
            x_vals=[bench_shape] if bench_shape is not None else REAL_SHAPES,
            line_arg="provider",
            line_vals=[triton_provider],
            line_names=[triton_provider],
            plot_name=f"triton_gmm_perf_{dtypes_desc}",
            args={},
            ylabel="TFLOPS",
        )
    )
    def benchmark(M: int, K: int, N: int, G: int, provider: str):
        assert "triton" in provider, f"GMM provider isn't triton, it's {provider}."

        logging.info("    (M, K, N, G) = (%d, %d, %d, %d)", M, K, N, G)

        lhs, rhs, group_sizes_0 = gen_input(
            M,
            K,
            N,
            G,
            preferred_element_type=in_dtype,
            trans_lhs=trans_lhs,
            trans_rhs=trans_rhs,
            rng_seed=rng_seed,
        )
        out = gen_output(M, N, preferred_element_type=out_dtype, trans=trans_out)
        multiple_group_sizes = gen_multiple_group_sizes(
            num_group_sizes, M, G, rng_seed=None, group_sizes_0=group_sizes_0
        )

        # First "dry-run" to invoke autotuner.
        triton_gmm(
            lhs, rhs, group_sizes_0, preferred_element_type=out_dtype, existing_out=out
        )

        quantiles = [0.5, 0.2, 0.8]
        p50_s_sum = 0.0
        p20_s_sum = 0.0
        p80_s_sum = 0.0
        tops_sum = 0.0

        for group_sizes in multiple_group_sizes:
            logging.debug(
                "      group_sizes (first 5) = %s", str(group_sizes[:5].tolist())
            )

            p50_ms, p20_ms, p80_ms = triton.testing.do_bench(
                lambda: triton_gmm(
                    lhs,
                    rhs,
                    group_sizes,
                    preferred_element_type=out_dtype,
                    existing_out=out,
                ),
                quantiles=quantiles,
            )
            p50_s_sum += 1e-3 * p50_ms
            p20_s_sum += 1e-3 * p20_ms
            p80_s_sum += 1e-3 * p80_ms
            tops_sum += torch.sum(1e-12 * 2 * group_sizes * N * K).item()

        p50_tflops = round(tops_sum / p50_s_sum, 2)
        p20_tflops = round(tops_sum / p80_s_sum, 2)
        p80_tflops = round(tops_sum / p20_s_sum, 2)

        logging.info(
            "      TFLOPS: p20 = %.2f, p50 = %.2f, p80 = %.2f",
            p20_tflops,
            p50_tflops,
            p80_tflops,
        )
        logging.info(
            "      best_config = %s", str(triton_autotuned_gmm_kernel.best_config)
        )

        return p50_tflops, p20_tflops, p80_tflops

    logging.info("Benchmarking Triton GMM kernel:")
    num_configs = len(autotune_configs())
    if num_configs > 50:
        logging.warning(
            "Warning: using full tuning space, there are %d configurations.",
            num_configs,
        )
    logging.info(
        "  input_type = %s, output_type = %s, num_group_sizes = %d",
        in_dtype_str,
        out_dtype_str,
        num_group_sizes,
    )
    logging.info(
        "  trans_lhs = %s, trans_rhs = %s, trans_out = %s",
        trans_lhs,
        trans_rhs,
        trans_out,
    )
    benchmark.run(show_plots=False, print_data=True)


# Standalone kernel runner.
# It's useful for `rocprof` profiling and collecting ATT traces.
# ------------------------------------------------------------------------------


def run_triton_gmm(
    M: int,
    K: int,
    N: int,
    G: int,
    in_dtype: torch.dtype = DTYPE,
    out_dtype: torch.dtype = DTYPE,
    trans_lhs: bool = TRANS_LHS,
    trans_rhs: bool = TRANS_RHS,
    trans_out: bool = TRANS_OUT,
    rng_seed: int = RNG_SEED,
    num_group_sizes: int = NUM_GROUP_SIZES,
) -> None:
    logging.info("Running Triton GMM kernel:")
    logging.info(
        "  input_type = %s, output_type = %s, num_group_sizes = %d",
        str_from_dtype(in_dtype),
        str_from_dtype(out_dtype),
        num_group_sizes,
    )
    logging.info(
        "  trans_lhs = %s, trans_rhs = %s, trans_out = %s",
        trans_lhs,
        trans_rhs,
        trans_out,
    )
    logging.info("    (M, K, N, G) = (%d, %d, %d, %d)", M, K, N, G)

    lhs, rhs, group_sizes_0 = gen_input(
        M,
        K,
        N,
        G,
        preferred_element_type=in_dtype,
        trans_lhs=trans_lhs,
        trans_rhs=trans_rhs,
        rng_seed=rng_seed,
    )
    multiple_group_sizes = gen_multiple_group_sizes(
        num_group_sizes, M, G, rng_seed=None, group_sizes_0=group_sizes_0
    )
    out = gen_output(M, N, preferred_element_type=out_dtype, trans=trans_out)

    for group_sizes in multiple_group_sizes:
        logging.debug("      group_sizes (first 5) = %s", str(group_sizes[:5].tolist()))
        triton_gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out,
            tiling=TILING,
        )


# Command line interface parsing.
# ------------------------------------------------------------------------------


def positive_int(value: str) -> int:
    try:
        int_value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    if int_value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
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

    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run GMM Triton kernel")

    # Shape
    parser.add_argument("M", type=positive_int, nargs="?", help="number of rows")
    parser.add_argument("K", type=positive_int, nargs="?", help="shared dimension")
    parser.add_argument("N", type=positive_int, nargs="?", help="number of columns")
    parser.add_argument("G", type=positive_int, nargs="?", help="number of groups")

    # Type
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

    # Transpose
    add_trans_arg(parser, "lhs", TRANS_LHS)
    add_trans_arg(parser, "rhs", TRANS_RHS)
    add_trans_arg(parser, "out", TRANS_OUT)

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


# Main function: entry point.
# ------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s > %(message)s",
        level=logging.INFO if args.verbose else logging.ERROR,
    )
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL + 1)

    shape = (args.M, args.K, args.N, args.G)
    in_dtype = dtype_from_str(args.input_type)
    out_dtype = dtype_from_str(args.output_type)

    if args.bench:
        benchmark_triton_gmm(
            bench_shape=None if all(arg is None for arg in shape) else shape,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            trans_lhs=args.trans_lhs,
            trans_rhs=args.trans_rhs,
            trans_out=args.trans_out,
            rng_seed=args.rng_seed,
            num_group_sizes=args.num_group_sizes,
        )
    else:
        run_triton_gmm(
            *shape,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            trans_lhs=args.trans_lhs,
            trans_rhs=args.trans_rhs,
            trans_out=args.trans_out,
            rng_seed=args.rng_seed,
            num_group_sizes=args.num_group_sizes,
        )


if __name__ == "__main__":
    main()
