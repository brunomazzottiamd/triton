import argparse
import csv
import dataclasses
import glob
import os


@dataclasses.dataclass
class ProfStatsFiles:
    python: str
    mojo: str


@dataclasses.dataclass
class ProfStatsTimeUs:
    python: float
    mojo: float


def profiling_dir() -> str:
    return os.path.join(os.getcwd(), "profiling")


def profiling_results_dir() -> str:
    results_dir = os.path.join(os.getcwd(), "profiling_results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    return results_dir


def list_lang_prof_stats_files(lang: str, kernel: str) -> dict[tuple[int, ...], str]:
    prof_dir = os.path.join(profiling_dir(), lang, kernel)
    glob_pattern = os.path.join(prof_dir, "*", "prof_stats.csv")
    shape_to_prof_stats_file = {}
    for prof_stats_file in glob.glob(glob_pattern):
        rel_path = os.path.relpath(prof_stats_file, prof_dir)
        shape_str = rel_path.split(os.path.sep)[0]
        shape = tuple(int(shape_dim) for shape_dim in shape_str.split("_"))
        shape_to_prof_stats_file[shape] = prof_stats_file
    assert all(
        os.path.exists(prof_stats_file)
        for prof_stats_file in shape_to_prof_stats_file.values()
    )
    return shape_to_prof_stats_file


def list_prof_stats_files(kernel: str) -> dict[tuple[int, ...], ProfStatsFiles]:
    python_prof_stats = list_lang_prof_stats_files("python", kernel)
    mojo_prof_stats = list_lang_prof_stats_files("mojo", kernel)
    common_shapes = python_prof_stats.keys() & mojo_prof_stats.keys()
    return {
        shape: ProfStatsFiles(
            python=python_prof_stats[shape], mojo=mojo_prof_stats[shape]
        )
        for shape in common_shapes
    }


def read_time_us_from_prof_stats_file(prof_stats_file: str) -> float | None:
    assert os.path.exists(prof_stats_file)
    with open(prof_stats_file, newline="") as csv_fd:
        csv_rows = list(csv.DictReader(csv_fd))
    return (
        float(csv_rows[0]["mean_us"])
        if len(csv_rows) == 1 and "mean_us" in csv_rows[0]
        else None
    )


def read_time_from_prof_stats_files(
    shape_to_prof_stats_files: dict[tuple[int, ...], ProfStatsFiles],
) -> dict[tuple[int, ...], ProfStatsTimeUs]:
    shape_to_prof_stats_times = {}
    for shape, prof_stats_files in shape_to_prof_stats_files.items():
        python_time_us = read_time_us_from_prof_stats_file(prof_stats_files.python)
        if python_time_us is not None:
            mojo_time_us = read_time_us_from_prof_stats_file(prof_stats_files.mojo)
            if mojo_time_us is not None:
                shape_to_prof_stats_times[shape] = ProfStatsTimeUs(
                    python=python_time_us, mojo=mojo_time_us
                )
    return shape_to_prof_stats_times


def prof_stats_time_to_csv_row(
    shape: tuple[int, ...], prof_stats_time: ProfStatsTimeUs
) -> dict[str, int | float]:
    return {f"shape_{i}": shape_dim for i, shape_dim in enumerate(shape)} | {
        "python_time_us": prof_stats_time.python,
        "mojo_time_us": prof_stats_time.mojo,
    }


def prof_stats_times_to_csv_data(
    shape_to_prof_stats_times: dict[tuple[int, ...], ProfStatsTimeUs],
) -> list[dict[str, int | float]]:
    return [
        prof_stats_time_to_csv_row(shape, shape_to_prof_stats_times[shape])
        for shape in sorted(shape_to_prof_stats_times.keys())
    ]


def write_perf_to_csv_file(
    shape_to_prof_stats_times: dict[tuple[int, ...], ProfStatsTimeUs], csv_file: str
) -> None:
    shape_len = max(len(shape) for shape in shape_to_prof_stats_times)
    csv_header = [f"shape_{i}" for i in range(0, shape_len)] + [
        "python_time_us",
        "mojo_time_us",
    ]
    csv_data = prof_stats_times_to_csv_data(shape_to_prof_stats_times)
    with open(csv_file, mode="w", newline="") as csv_fd:
        csv_writer = csv.DictWriter(csv_fd, fieldnames=csv_header)
        csv_writer.writeheader()
        csv_writer.writerows(csv_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agg_profiling", description="aggregate kernel profiling data"
    )
    parser.add_argument(
        "-k", "--kernel", required=True, help="kernel to aggregate profiling data"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output CSV file, defaults to profiling_results_dir/kernel.csv",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(profiling_results_dir(), f"{args.kernel}.csv")
    return args


def main():
    args = parse_args()
    write_perf_to_csv_file(
        read_time_from_prof_stats_files(list_prof_stats_files(args.kernel)),
        args.output,
    )


if __name__ == "__main__":
    main()
