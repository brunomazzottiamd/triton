#!/usr/bin/env python


# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

import argparse
from datetime import datetime, timedelta
import logging
import os
import re
import sys
from typing import Any
from zipfile import ZipFile

import pandas as pd


# Get results from `bench_gmm.sh`.
# ------------------------------------------------------------------------------


def is_valid_file(file_name: str) -> bool:
    return (
        os.path.exists(file_name)
        and os.path.isfile(file_name)
        and os.access(file_name, os.R_OK)
    )


def get_bench_metadata(
    zip_file: ZipFile,
) -> list[tuple[str, int, int, int, int, str, str]]:
    bench_file_name_pattern: re.Pattern = re.compile(
        r"^bench_(\d+)_(\d+)_(\d+)_(\d+)_(.+)_([cr]{3})\.log$"
    )

    bench_metadata: list[tuple[str, int, int, int, int, str, str]] = []

    for file_name in zip_file.namelist():
        bench_file_name_match: re.Match[str] | None = bench_file_name_pattern.fullmatch(
            file_name
        )

        if bench_file_name_match:
            logging.debug(
                "Found [%s] benchmark file in [%s].", file_name, zip_file.filename
            )

            m: int = int(bench_file_name_match.group(1))
            k: int = int(bench_file_name_match.group(2))
            n: int = int(bench_file_name_match.group(3))
            g: int = int(bench_file_name_match.group(4))
            kernel: str = bench_file_name_match.group(5).upper()
            layout: str = bench_file_name_match.group(6)

            if layout == "rrr":
                layout = "NN"
            elif layout == "rcr":
                layout = "NT"
            elif layout == "crr":
                layout = "TN"
            else:
                layout = layout.upper()

            bench_metadata.append((file_name, m, k, n, g, kernel, layout))

    if not bench_metadata:
        logging.error("There's no benchmark files in [%s].", zip_file.filename)
    else:
        logging.info(
            "Found %d benchmark files in [%s].", len(bench_metadata), zip_file.filename
        )

    return bench_metadata


TFLOPS_PATTERN: re.Pattern = re.compile(
    r"TFLOPS: p20 =\s*(\d+\.\d{2}), p50 =\s*(\d+\.\d{2}), p80 =\s*(\d+\.\d{2})",
    re.MULTILINE,
)


def get_tflops(
    bench_file_name: str, bench_file_content: str
) -> tuple[float, float, float] | None:
    tflops_match: re.Match[str] | None = TFLOPS_PATTERN.search(bench_file_content)

    if not tflops_match:
        logging.warning(
            "Could not find TFLOPS data in [%s], skipping it.",
            bench_file_name,
        )
        return None

    p20_tflops: float = float(tflops_match.group(1))
    p50_tflops: float = float(tflops_match.group(2))
    p80_tflops: float = float(tflops_match.group(3))

    if not (p80_tflops >= p50_tflops >= p20_tflops):
        logging.warning(
            "TFLOPS data in [%s] seems to be malformed, skipping it.",
            bench_file_name,
        )
        return None

    logging.debug(
        "TFLOPS in [%s]: %.2f, %.2f, %.2f",
        bench_file_name,
        p20_tflops,
        p50_tflops,
        p80_tflops,
    )

    return p20_tflops, p50_tflops, p80_tflops


BEST_CONFIG_PATTERN: re.Pattern = re.compile(r"best_config = (.+)", re.MULTILINE)


def get_best_config(
    bench_file_name: str, bench_file_content: str
) -> dict[str, int] | None:
    best_config_match: re.Match[str] | None = BEST_CONFIG_PATTERN.search(
        bench_file_content
    )

    if not best_config_match:
        logging.warning(
            "Could not find best config data in [%s], skipping it.",
            bench_file_name,
        )
        return None

    best_config: dict[str, int] = {
        key.strip(): int(value.strip())
        for key, value in (
            key_value.split(":")
            for key_value in best_config_match.group(1).split(",")
            if not ("num_ctas" in key_value or "maxnreg" in key_value)
        )
    }

    logging.debug("Best config in [%s]: %s", bench_file_name, best_config)

    return best_config


TUNING_CONFIGS_PATTERN: re.Pattern = re.compile(r"there are (\d+) configurations\.")


# Get number of distinct tuning configs.
def get_num_tuning_configs(bench_file_name: str, bench_file_content: str) -> int | None:
    tuning_configs_match: re.Match[str] | None = TUNING_CONFIGS_PATTERN.search(
        bench_file_content
    )

    if not tuning_configs_match:
        logging.warning(
            "Could not find number of tuning configs in [%s], skipping it.",
            bench_file_name,
        )
        return None

    tuning_configs: int = int(tuning_configs_match.group(1))
    logging.debug("Tuning performed with %d configs.", tuning_configs)

    return tuning_configs


TIMESTAMP_PATTERN: re.Pattern = re.compile(
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", re.MULTILINE
)

TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M:%S,%f"


# Get total tuning time in hours.
def get_tuning_time_hours(
    bench_file_name: str, bench_file_content: str
) -> float | None:
    timestamps = re.findall(TIMESTAMP_PATTERN, bench_file_content)

    if len(timestamps) < 2:
        logging.warning(
            "Not enough timestamps found in [%s], skipping it.",
            bench_file_name,
        )
        return None

    start_timestamp: datetime = datetime.strptime(timestamps[0], TIMESTAMP_FORMAT)
    logging.debug("Tuning started at %s.", start_timestamp)
    end_timestamp: datetime = datetime.strptime(timestamps[-1], TIMESTAMP_FORMAT)
    logging.debug("Tuning ended at %s.", end_timestamp)

    if end_timestamp < start_timestamp:
        logging.error(
            "Inconsistent tuning timestamps in [%s], skipping it.", bench_file_name
        )
        return None

    elapsed_time: timedelta = end_timestamp - start_timestamp
    logging.debug("Elapsed tuning time is %s.", elapsed_time)

    return round(elapsed_time.total_seconds() / 3600, 2)


def get_bench_results(zip_file_name: str) -> pd.DataFrame | None:
    logging.info("Processing [%s] file...", zip_file_name)

    if not is_valid_file(zip_file_name):
        logging.error("[%s] isn't a valid file.", zip_file_name)
        return None

    bench_data: list[dict[str, Any]] = []

    with ZipFile(zip_file_name, "r") as zip_file:
        for bench_file_name, m, k, n, g, kernel, layout in get_bench_metadata(zip_file):
            with zip_file.open(bench_file_name) as bench_file:
                bench_file_content: str = bench_file.read().decode("utf-8")

                tflops: tuple[float, float, float] | None = get_tflops(
                    bench_file_name, bench_file_content
                )
                if tflops is None:
                    continue

                best_config: dict[str, int] | None = get_best_config(
                    bench_file_name, bench_file_content
                )
                if best_config is None:
                    continue

                num_tuning_configs: int | None = get_num_tuning_configs(
                    bench_file_name, bench_file_content
                )
                if num_tuning_configs is None:
                    continue

                tuning_time_hours: float | None = get_tuning_time_hours(
                    bench_file_name, bench_file_content
                )
                if tuning_time_hours is None:
                    continue

                bench_data.append(
                    {
                        "M": m,
                        "K": k,
                        "N": n,
                        "G": g,
                        "Kernel": kernel,
                        "Layout": layout,
                        "TFLOPS": tflops[1],
                        "Number of Tuning Configs": num_tuning_configs,
                        "Total Tuning Time (h)": tuning_time_hours,
                    }
                    | best_config
                )

    if not bench_data:
        logging.error("There's no valid data in [%s].", zip_file_name)
        return None

    df: pd.DataFrame = pd.DataFrame(bench_data)
    df["Tuning Time per Config (s)"] = (
        (3600 * df["Total Tuning Time (h)"]) / df["Number of Tuning Configs"]
    ).round(2)
    return df


def print_dataframe(df: pd.DataFrame, format: str) -> None:
    assert format in {"md", "csv"}
    if format == "md":
        print(df.to_markdown(index=False))
    else:
        df.to_csv(sys.stdout, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="extract data from GMM benchmark zip file"
    )
    parser.add_argument("zip_file", help="zip file to process")
    parser.add_argument(
        "-f", "--format", choices=["md", "csv"], default="md", help="output format"
    )
    return parser.parse_args()


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

    id_cols = ["M", "K", "N", "G", "Kernel", "Layout"]
    bench_data.sort_values(by=id_cols, inplace=True)

    logging.info("Performance:")
    perf_cols = ["TFLOPS"]
    print_dataframe(bench_data[id_cols + perf_cols], args.format)

    logging.info("Tuning time:")
    tuning_time_cols = [
        "Number of Tuning Configs",
        "Total Tuning Time (h)",
        "Tuning Time per Config (s)",
    ]
    print_dataframe(bench_data[id_cols + tuning_time_cols], args.format)

    logging.info("Best tuning configuration:")
    print_dataframe(bench_data.drop(columns=perf_cols + tuning_time_cols), args.format)


if __name__ == "__main__":
    main()
