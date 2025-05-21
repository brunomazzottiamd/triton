#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

black "${script_dir}"/*.py
ruff check "${script_dir}"/*.py

# mypy has issues with Triton on Windows, run it only on Linux.
if [ "$(uname)" == Linux ]; then
    mypy --ignore-missing-imports "${script_dir}"/*.py
fi

if [ "$(uname)" == Linux ]; then
    shellcheck "${script_dir}"/*.sh
else
    # Ignore literal carriage return warnings on Windows.
    shellcheck --exclude SC1017 "${script_dir}"/*.sh
fi
