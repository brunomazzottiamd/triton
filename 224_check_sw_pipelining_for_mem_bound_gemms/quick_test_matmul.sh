#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

test_matmul() {
    m="${1}"
    n="${2}"
    k="${3}"
    echo "Testing 'matmul' for (M, N, K) = (${m}, ${n}, ${k})"
    python "${script_dir}/test_matmul.py" -m "${m}" -n "${n}" -k "${k}"
}

declare -a shapes=(
    '1 8192 28672'
    '1 6144 6144'
    '1 4096 4096'
    '2 16384 16384'
)

for shape in "${shapes[@]}"; do
    read -ra mnk <<< "${shape}"
    test_matmul "${mnk[@]}"
done
