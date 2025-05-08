#!/usr/bin/env bash

py_src=(gmm_kernel.py gmm.py)

black "${py_src[@]}"
ruff check "${py_src[@]}"

# mypy has issues with Tirton on Windows...
if [ "$(uname)" == Linux ]; then
    mypy --ignore-missing-imports "${py_src[@]}"
fi
