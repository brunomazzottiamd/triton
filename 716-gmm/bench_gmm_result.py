#!/usr/bin/env python


# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

import argparse
import logging
import os
import re
from typing import Any
import zipfile

import pandas as pd


# Get results from `bench_gmm.sh`.
# ------------------------------------------------------------------------------


def is_valid_file(file_name: str) -> bool:
    return (
        os.path.exists(file_name)
        and os.path.isfile(file_name)
        and os.access(file_name, os.R_OK)
    )


def get_bench_results(zip_file_name: str) -> pd.DataFrame | None:
    logging.info("Processing [%s] file...", zip_file_name)

    if not is_valid_file(zip_file_name):
        logging.error("[%s] isn't a valid file.", zip_file_name)
        return None

    bench_data: list[dict[str, Any]] = []

    with zipfile.ZipFile(zip_file_name, "r") as zip_file:
        bench_file_name_pattern: re.Pattern = re.compile(
            r"^bench_(\d+)_(\d+)_(\d+)_(\d+)_([cr]{3})\.log$"
        )

        bench_metadata: list[tuple[str, int, int, int, int, str]] = []

        for file_name in zip_file.namelist():
            bench_file_name_match: re.Match[str] | None = (
                bench_file_name_pattern.fullmatch(file_name)
            )
            if bench_file_name_match:
                logging.debug(
                    "Found [%s] benchmark file in [%s].", file_name, zip_file_name
                )

                m: int = int(bench_file_name_match.group(1))
                k: int = int(bench_file_name_match.group(2))
                n: int = int(bench_file_name_match.group(3))
                g: int = int(bench_file_name_match.group(4))
                layout: str = bench_file_name_match.group(5)
                if layout == "rcr":
                    layout = "TN"
                elif layout == "ccr":
                    layout = "NN"
                elif layout == "crr":
                    layout = "NT"
                bench_metadata.append((file_name, m, k, n, g, layout))

        if not bench_metadata:
            logging.error("There's no benchmark files in [%s].", zip_file_name)
            return None

        logging.info(
            "Found %d benchmark files in [%s].", len(bench_metadata), zip_file_name
        )

        tflops_pattern: re.Pattern = re.compile(
            r"TFLOPS: p20 = (\d+\.\d{2}), p50 = (\d+\.\d{2}), p80 = (\d+\.\d{2})",
            re.MULTILINE,
        )

        best_config_pattern: re.Pattern = re.compile(
            r"best_config = (.+)", re.MULTILINE
        )

        for bench_file_name, m, k, n, g, layout in bench_metadata:
            with zip_file.open(bench_file_name) as bench_file:
                bench_file_content: str = bench_file.read().decode("utf-8")

                tflops_match: re.Match[str] | None = tflops_pattern.search(
                    bench_file_content
                )
                if not tflops_match:
                    logging.warning(
                        "Could not find TFLOPS data in [%s], skipping it.",
                        bench_file_name,
                    )
                    continue
                p20_tflops: float = float(tflops_match.group(1))
                p50_tflops: float = float(tflops_match.group(2))
                p80_tflops: float = float(tflops_match.group(3))
                if not (p80_tflops >= p50_tflops >= p20_tflops):
                    logging.warning(
                        "TFLOPS data in [%s] seems to be malformed, skipping it.",
                        bench_file_name,
                    )
                    continue
                logging.debug(
                    "TFLOPS in [%s]: %.2f, %.2f, %.2f",
                    bench_file_name,
                    p20_tflops,
                    p50_tflops,
                    p80_tflops,
                )

                best_config_match: re.Match[str] | None = best_config_pattern.search(
                    bench_file_content
                )
                if not best_config_match:
                    logging.warning(
                        "Could not find best config data in [%s], skipping it.",
                        bench_file_name,
                    )
                    continue
                best_config: dict[str, int] = {
                    key.strip(): int(value.strip())
                    for key, value in (
                        key_value.split(":")
                        for key_value in best_config_match.group(1).split(",")
                        if not ("num_ctas" in key_value or "maxnreg" in key_value)
                    )
                }
                logging.debug("Best config in [%s]: %s", bench_file_name, best_config)

                bench_data.append(
                    {
                        "M": m,
                        "K": k,
                        "N": n,
                        "G": g,
                        "Layout": layout,
                        "TFLOPS": p50_tflops,
                    }
                    | best_config
                )

    if not bench_data:
        logging.error("There's no valid data in [%s].", zip_file_name)
        return None

    return pd.DataFrame(bench_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="extract data from GMM benchmark zip file"
    )
    parser.add_argument("zip_file", help="zip file to process")
    return parser.parse_args()


def print_markdown(df: pd.DataFrame) -> None:
    print(df.to_markdown(index=False))


def main() -> None:
    args: argparse.Namespace = parse_args()

    logging.basicConfig(format="%(asctime)s > %(message)s", level=logging.INFO)

    try:
        bench_data: pd.DataFrame | None = get_bench_results(args.zip_file)
    except Exception as error:
        logging.error("Unexpected error: %s", error)
        return

    if bench_data is None:
        return

    logging.info("Performance:")
    print_markdown(bench_data[["M", "K", "N", "G", "Layout", "TFLOPS"]])

    logging.info("Best tuning configuration:")
    print_markdown(bench_data.drop(columns=["TFLOPS"]))


if __name__ == "__main__":
    main()
