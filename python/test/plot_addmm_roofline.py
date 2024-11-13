#!/usr/bin/env python

# -*- coding: utf-8 -*-

import pandas as pd


def get_target_shapes() -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame(
        [
            ("Shape #1", 20196, 512, 1536),
            ("Shape #2", 171792, 512, 1536),
            ("Shape #3", 173318, 512, 1536),
        ],
        columns=["desc", "M", "N", "K"],
    )
    return df


def compute_arith_intensity(df: pd.DataFrame) -> pd.DataFrame:
    M: pd.Series = df["M"]
    N: pd.Series = df["N"]
    K: pd.Series = df["K"]
    ops: pd.Series = 2 * M * N * K
    bytes: pd.Series = 2 * M * K + 2 * K * N + 4 * M * N
    df["arith_intensity"] = ops / bytes
    return df


def get_target_hardware_specs() -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame(
        [
            ("Peak MI300X", 1307.4, 5.3),
            ("Achievable MI300X", 650, 3.5),
            ("Peak H100 PCIe", 1513, 2),
            ("Peak H100 SXM", 1979, 3.35),
        ],
        columns=["desc", "comp_perf_tops", "mem_bandwidth_tbytes"],
    )
    return df


def compute_ops_bytes_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df["ops_bytes_ratio"] = df["comp_perf_tops"] / df["mem_bandwidth_tbytes"]
    return df


def main() -> None:
    shapes_df: pd.DataFrame = compute_arith_intensity(get_target_shapes())
    print(shapes_df)
    hardware_df: pd.DataFrame = compute_ops_bytes_ratio(get_target_hardware_specs())
    print(hardware_df)


if __name__ == "__main__":
    main()
