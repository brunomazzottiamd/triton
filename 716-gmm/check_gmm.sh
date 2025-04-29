#!/usr/bin/env bash

black gmm.py
ruff check gmm.py
mypy --ignore-missing-imports gmm.py
