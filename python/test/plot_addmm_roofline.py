#!/usr/bin/env python

# -*- coding: utf-8 -*-

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd


def get_target_shapes() -> pd.DataFrame:
    # Basic data: textual description, M, N, K
    df: pd.DataFrame = pd.DataFrame(
        [
            # Shapes of `addmm` kernel:
            # ("#1", 20196, 512, 1536),
            # ("#2", 171792, 512, 1536),
            # ("#3", 173318, 512, 1536),
            # Shapes of `tem_fused_addmm_130` kernel:
            ("", 84122, 2048, 256)
        ],
        columns=["desc", "M", "N", "K"],
    )
    # Compute arithmetic intensity:
    M: pd.Series = df["M"]
    N: pd.Series = df["N"]
    K: pd.Series = df["K"]
    ops: pd.Series = 2 * M * N * K
    # Y with (M, N) shape:
    # bytes: pd.Series = 2 * M * K + 2 * K * N + 4 * M * N
    # Y with (1, N) shape + broadcasting:
    bytes: pd.Series = 2 * M * K + 2 * K * N + 2 * N + 2 * M * N
    df["arith_intensity"] = ops / bytes
    return df


def get_target_hardware_specs() -> pd.DataFrame:
    # Basic data: textual description, compute performance in TOp/s, memory bandwidth in TB/s
    df: pd.DataFrame = pd.DataFrame(
        [
            ("Peak MI300X", 1307.4, 5.3),
            ("Achievable MI300X", 650, 3.5),
            # Dividing Nvidia performance numbers by 2 because the reported numbers use 2:4 sparsity.
            ("Peak H100 PCIe", 1513 / 2, 2),
            ("Peak H100 SXM", 1979 / 2, 3.35),
        ],
        columns=["desc", "comp_perf_tops", "mem_bandwidth_tbs"],
    )
    # Compute arithmetic ops : bytes ratio:
    # We can divide without converting units because both compute and memory capabilities are
    # expressed with Tera prefix.
    df["ops_bytes_ratio"] = df["comp_perf_tops"] / df["mem_bandwidth_tbs"]
    return df


class ColorPalette:
    colors: list[str]
    color_index: int

    def __init__(self):
        # Categorical color palette from this website:
        # https://carbondesignsystem.com/data-visualization/color-palettes/#categorical-palettes
        self.colors = [
            "#6929c4",  # Purple 70
            "#1192e8",  # Cyan 50
            "#005d5d",  # Teal 70
            "#9f1853",  # Magenta 70
            "#fa4d56",  # Red 50
            "#570408",  # Red 90
            "#198038",  # Green 60
            "#002d9c",  # Blue 80
            "#ee538b",  # Magenta 50
            "#b28600",  # Yellow 50
            "#009d9a",  # Teal 50
            "#012749",  # Cyan 90
            "#8a3800",  # Orange 70
            "#a56eff",  # Purple 50
        ]
        self.color_index = 0

    def next_color(self) -> str:
        color: str = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return color


def main() -> None:
    # Get data:
    shapes_df: pd.DataFrame = get_target_shapes()
    hw_df: pd.DataFrame = get_target_hardware_specs()

    # Set chart size:
    fig_width: float
    fig_height: float
    fig_width, fig_height = plt.rcParams["figure.figsize"]
    fig_scale: float = math.sqrt(2)
    fig_width *= fig_scale
    fig_height *= fig_scale
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    # Set chart title and axes' titles:
    kernel_name: str
    # kernel_name = "addmm"
    kernel_name = "tem_fused_addmm_130"
    ax.set(
        title=f"Phantom {kernel_name} Kernel Roofline",
        xlabel="Arithmetic Intensity (Op / B)",
        ylabel="Performance (TOp / s)",
    )

    # X-axis is arithmetic intensity:
    max_x: float = 1.05 * max(shapes_df["arith_intensity"].max(), hw_df["ops_bytes_ratio"].max())
    x: npt.NDArray = np.arange(0, max_x, 0.1)

    # Get color palette:
    color_palette: ColorPalette = ColorPalette()

    # Plot rooflines for each hardware:
    rooflines: list = []
    for _, hw in hw_df.iterrows():
        y: npt.NDArray = np.minimum(hw["comp_perf_tops"], hw["mem_bandwidth_tbs"] * x)
        color: str = color_palette.next_color()
        roofline, = ax.plot(x, y, label=hw["desc"], color=color)
        rooflines.append(roofline)
        ax.vlines(
            x=hw["ops_bytes_ratio"],
            ymin=0,
            ymax=hw["comp_perf_tops"],
            color=color,
            linestyles="dotted",
        )

    # Plot legend for rooflines:
    rooflines_legend = ax.legend(handles=rooflines, title="Hardware Rooflines", loc="upper left",
                                 bbox_to_anchor=(1.05, 1))
    ax.add_artist(rooflines_legend)

    # Plot arithmetic intensity of each shape:
    max_y: float = ax.get_ylim()[1]
    shapes: list = []
    for _, shape in shapes_df.iterrows():
        label: str = f"{shape['desc']} ({shape['M']}, {shape['N']}, {shape['K']})"
        shape = ax.vlines(
            x=shape["arith_intensity"],
            ymin=0,
            ymax=max_y,
            label=label,
            color=color_palette.next_color(),
            linestyles="dashed",
        )
        shapes.append(shape)

    # Plot lengend for shapes:
    roofline_legend_bbox = rooflines_legend.get_window_extent().transformed(fig.transFigure.inverted())
    shapes_legend = ax.legend(
        handles=shapes,
        title="Shapes",
        loc="upper left",
        bbox_to_anchor=(1.05, roofline_legend_bbox.y0),
    )
    ax.add_artist(shapes_legend)

    # Save chart to image file:
    plot_img_file_name: str = os.path.splitext(os.path.basename(__file__))[0] + ".png"
    fig.savefig(plot_img_file_name)


if __name__ == "__main__":
    main()
