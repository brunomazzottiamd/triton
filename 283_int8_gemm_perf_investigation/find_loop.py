import sys
import json
import statistics
from pathlib import Path
from tabulate import tabulate


def find_loop_instructions(code_file_path: Path) -> tuple[int, int]:  # (loop_start_index, loop_end_index)
    # Get instructions from JSON code file.
    with open(code_file_path, "r") as code_file:
        code: list[str] = [code_line[0] for code_line in json.load(code_file)["code"]]

    # Search for loop start instruction.
    loop_start_inst: str = "v_lshl_add_u64 v[100:101], v[24:25], 0, s[0:1]"
    try:
        loop_start_index: int = code.index(loop_start_inst)
    except ValueError:
        loop_start_index = -1

    # Search for loop end instruction.
    # It's always a "s_cbranch_scc1" but address changes from kernel to kernel.
    loop_end_inst: str = "s_cbranch_scc1 65"
    try:
        loop_end_index: int = next(
            len(code) - code_line_index - 1
            for code_line_index, code_line in enumerate(reversed(code))
            if code_line.startswith(loop_end_inst))
    except StopIteration:
        loop_end_index = -1

    return loop_start_index, loop_end_index


def find_loop_boundaries(
        wave_file_path: Path, loop_start_index: int,
        loop_end_index: int) -> tuple[int, int, int]:  # (loop_start_clock, loop_end_clock, loop_cycles)
    # Get instructions from JSON wave file.
    with open(wave_file_path, "r") as wave_file:
        wave_instructions: list[tuple[int, int]] = [
            (wave_instructions[0], wave_instructions[4])  # (clock, code_line)
            for wave_instructions in json.load(wave_file)["wave"]["instructions"]
            if wave_instructions[4] == loop_start_index or wave_instructions[4] == loop_end_index
        ]

    # Get first occurence of loop start.
    loop_start_clock: int = next(clock for clock, code_line in wave_instructions if code_line == loop_start_index)

    # Get last occurence of loop end.
    loop_end_clock: int = next(clock for clock, code_line in reversed(wave_instructions) if code_line == loop_end_index)

    # Compute loop cycles.
    assert loop_end_clock > loop_start_clock
    loop_cycles: int = loop_end_clock - loop_start_clock + 1

    return loop_start_clock, loop_end_clock, loop_cycles


def main():
    ui_dir: Path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

    loop_start_index: int
    loop_end_index: int
    loop_start_index, loop_end_index = find_loop_instructions(ui_dir / "code.json")

    if loop_start_index < 0 or loop_end_index < 0:
        print("Unable to find loop instructions.", file=sys.stderr)
        return

    loop_boundaries: list[tuple[str, int, int, int]] = [
        (wave_file_path.stem, ) + find_loop_boundaries(wave_file_path, loop_start_index, loop_end_index)
        for wave_file_path in ui_dir.glob("se*_sm*_sl*_wv*.json")
    ]

    loop_cycles: list[int] = [loop_boundary[-1] for loop_boundary in loop_boundaries]
    mean_loop_cycles: float = round(statistics.mean(loop_cycles), 2)
    stdev_loop_cycles: float = round(statistics.stdev(loop_cycles), 2)

    print(f"ATT trace:\n{ui_dir}\n")
    print(tabulate(loop_boundaries, headers=("Wave", "Loop start", "Loop end", "Loop cycles")))
    print(f"Mean loop cycles: {mean_loop_cycles} ± {stdev_loop_cycles}")


if __name__ == "__main__":
    main()
