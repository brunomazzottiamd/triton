#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
python_dir="${script_dir}/python"

shellcheck --shell=bash "${script_dir}/"*.sh &&
    black --target-version py312 "${python_dir}"/*.py &&
    ruff check "${python_dir}"/*.py &&
    mypy --ignore-missing-imports "${python_dir}"/*.py
