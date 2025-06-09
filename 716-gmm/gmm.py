#!/usr/bin/env python


# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import logging

# PyTorch
import torch

# Triton
import triton

# Types module
from dtypes import (
    DTYPE,
    dtype_from_str,
    str_from_dtype,
)

# Common module
from common import (
    RNG_SEED,
    NUM_GROUP_SIZES,
    REAL_SHAPES,
)
from gmm_common import gen_gmm_tensors
from tgmm_common import gen_tgmm_tensors

# Triton GMM implementation
from triton_gmm import triton_gmm, gmm_autotune_configs, triton_gmm_kernel_autotuned

# Triton TGMM implementation (persistent only for now)
from triton_tgmm import (
    triton_persistent_tgmm,
    tgmm_persistent_autotune_configs,
    triton_tgmm_persistent_autotuned_kernel,
    triton_non_persistent_tgmm,
    tgmm_non_persistent_autotune_configs,
    triton_tgmm_non_persistent_autotuned_kernel,
)

# CLI parser
from cli_parser import parse_args


# Selection of different implementations.
# (used by benchmark and standalone runner)
# ------------------------------------------------------------------------------


def select_triton_kernel(gmm_type):
    assert gmm_type in {"gmm", "ptgmm", "tgmm"}, "Invalid GMM type."
    if gmm_type == "gmm":
        desc, gen_tensors, kernel_wrapper, kernel, autotune_configs = (
            "GMM",
            gen_gmm_tensors,
            triton_gmm,
            triton_gmm_kernel_autotuned,
            gmm_autotune_configs,
        )
    if gmm_type == "ptgmm":
        desc, gen_tensors, kernel_wrapper, kernel, autotune_configs = (
            "persistent TGMM",
            gen_tgmm_tensors,
            triton_persistent_tgmm,
            triton_tgmm_persistent_autotuned_kernel,
            tgmm_persistent_autotune_configs,
        )
    if gmm_type == "tgmm":
        desc, gen_tensors, kernel_wrapper, kernel, autotune_configs = (
            "non-persistent TGMM",
            gen_tgmm_tensors,
            triton_non_persistent_tgmm,
            triton_tgmm_non_persistent_autotuned_kernel,
            tgmm_non_persistent_autotune_configs,
        )
    return (
        desc,
        gen_tensors,
        kernel_wrapper,
        kernel,
        autotune_configs,
    )


# Benchmark.
# ------------------------------------------------------------------------------


def benchmark_triton(
    gmm_type: str,
    bench_shape: tuple[int, int, int, int] | None = None,
    in_dtype: torch.dtype = DTYPE,
    out_dtype: torch.dtype = DTYPE,
    rng_seed: int = RNG_SEED,
    num_group_sizes: int = NUM_GROUP_SIZES,
    unif_group_sizes: bool = False,
) -> None:
    desc, gen_tensors, kernel_wrapper, kernel, autotune_configs = select_triton_kernel(
        gmm_type
    )

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
            plot_name=f"triton_{gmm_type}_perf_{dtypes_desc}",
            args={},
            ylabel="TFLOPS",
        )
    )
    def benchmark(M: int, K: int, N: int, G: int, provider: str):
        assert "triton" in provider, f"Provider isn't triton, it's {provider}."

        logging.info("    (M, K, N, G) = (%d, %d, %d, %d)", M, K, N, G)

        lhs, rhs, multiple_group_sizes, out = gen_tensors(
            M,
            K,
            N,
            G,
            num_group_sizes,
            input_type=in_dtype,
            output_type=out_dtype,
            rng_seed=rng_seed,
            unif_group_sizes=unif_group_sizes,
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
                lambda: kernel_wrapper(
                    lhs,
                    rhs,
                    group_sizes,
                    preferred_element_type=out_dtype,
                    existing_out=out,
                    autotune=True,
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
            "      TFLOPS: p20 = %6.2f, p50 = %6.2f, p80 = %6.2f",
            p20_tflops,
            p50_tflops,
            p80_tflops,
        )

        best_config = (
            str(kernel.best_config)
            .replace(", num_ctas: 1", "")
            .replace(", maxnreg: None", "")
        )
        logging.info("      best_config = %s", best_config)

        return p50_tflops, p20_tflops, p80_tflops

    logging.info("Benchmarking Triton %s kernel:", desc)

    num_configs = len(autotune_configs())
    if num_configs > 50:  # this is a completely arbitrary threshold!
        logging.warning(
            "  Warning: using full tuning space, there are %d configurations.",
            num_configs,
        )
    else:
        logging.info(
            "  Using reduced tuning space, there are %d configurations.",
            num_configs,
        )

    logging.info(
        "  input_type = %s, output_type = %s, rng_seed = %d",
        in_dtype_str,
        out_dtype_str,
        rng_seed,
    )
    logging.info(
        "  num_group_sizes = %d, unif_group_sizes = %s",
        num_group_sizes,
        unif_group_sizes,
    )
    benchmark.run(show_plots=False, print_data=True)


# Standalone kernel runner.
# It's useful for `rocprof` profiling and collecting ATT traces.
# ------------------------------------------------------------------------------


def run_triton(
    gmm_type: str,
    M: int,
    K: int,
    N: int,
    G: int,
    in_dtype: torch.dtype = DTYPE,
    out_dtype: torch.dtype = DTYPE,
    rng_seed: int = RNG_SEED,
    num_group_sizes: int = NUM_GROUP_SIZES,
    unif_group_sizes: bool = False,
) -> None:
    desc, gen_tensors, kernel_wrapper, _, _ = select_triton_kernel(gmm_type)

    logging.info("Running Triton %s kernel:", desc)
    logging.info(
        "  input_type = %s, output_type = %s, rng_seed = %d",
        str_from_dtype(in_dtype),
        str_from_dtype(out_dtype),
        rng_seed,
    )
    logging.info(
        "  num_group_sizes = %d, unif_group_sizes = %s",
        num_group_sizes,
        unif_group_sizes,
    )
    logging.info("  (M, K, N, G) = (%d, %d, %d, %d)", M, K, N, G)

    lhs, rhs, multiple_group_sizes, out = gen_tensors(
        M,
        K,
        N,
        G,
        num_group_sizes,
        input_type=in_dtype,
        output_type=out_dtype,
        rng_seed=rng_seed,
        unif_group_sizes=unif_group_sizes,
    )

    for group_sizes in multiple_group_sizes:
        logging.debug("    group_sizes (first 5) = %s", str(group_sizes[:5].tolist()))
        kernel_wrapper(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out,
        )


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
        benchmark_triton(
            args.gmm_type,
            bench_shape=None if all(arg is None for arg in shape) else shape,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            rng_seed=args.rng_seed,
            num_group_sizes=args.num_group_sizes,
            unif_group_sizes=args.unif_group_sizes,
        )
    else:
        run_triton(
            args.gmm_type,
            *shape,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            rng_seed=args.rng_seed,
            num_group_sizes=args.num_group_sizes,
            unif_group_sizes=args.unif_group_sizes,
        )


if __name__ == "__main__":
    main()
