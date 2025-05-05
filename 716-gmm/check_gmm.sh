#!/usr/bin/env bash

black gmm.py
ruff check gmm.py
# mypy has issues with Tirton on Windows...
if [ "$(uname)" == Linux ]; then
    mypy --ignore-missing-imports gmm.py
fi
