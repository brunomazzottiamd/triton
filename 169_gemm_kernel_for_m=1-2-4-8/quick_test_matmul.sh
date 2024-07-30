#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
test_matmul_py="${script_dir}/test_matmul.py"

python "${test_matmul_py}" single -m 1 -n 8192 -k 8192
python "${test_matmul_py}" single -m 1 -n 6144 -k 6144
python "${test_matmul_py}" single -m 1 -n 4096 -k 4096
python "${test_matmul_py}" single -m 2 -n 16384 -k 16384
