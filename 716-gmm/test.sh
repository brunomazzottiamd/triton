#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

export TRITON_DEBUG=1
pytest --verbose --exitfirst -s --tb=no "${@}" "${script_dir}/test_gmm.py::test_tgmm"
