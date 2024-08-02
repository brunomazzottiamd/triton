#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

test_matmul() {
    dot="${1}"
    m="${2}"
    n="${3}"
    k="${4}"
    echo "Testing 'matmul' with '${dot}' dot product for (M, N, K) = (${m}, ${n}, ${k})"
    python "${script_dir}/test_matmul.py" \
        single --dot "${dot}" -m "${m}" -n "${n}" -k "${k}"
}

declare -a dots=(
    'tl_dot'
    'multreduce'
)

declare -a shapes=(
    '1 8192 28672'
    '1 6144 6144'
    '1 4096 4096'
    '2 16384 16384'
)

for dot in "${dots[@]}"; do
    for shape in "${shapes[@]}"; do
        read -ra mnk <<< "${shape}"
        test_matmul "${dot}" "${mnk[@]}"
    done
done
