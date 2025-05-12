#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

black "${script_dir}"/*.py
ruff check "${script_dir}"/*.py
# mypy has issues with Tirton on Windows...
if [ "$(uname)" == Linux ]; then
    mypy --ignore-missing-imports "${script_dir}"/*.py
fi

shellcheck "${script_dir}"/*.sh
