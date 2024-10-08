#!/usr/bin/env bash

echo 'Running hipBLASLt benchmark...'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
output_dir="${script_dir}/hipblaslt_results"
rm --recursive --force "${output_dir}"
mkdir --parents "${output_dir}"

declare -a target_shapes=(
    '20 1920 13312'
    '30 1920 13312'
    '20 17792 13312'
    '30 17792 13312'
)

for shape in "${target_shapes[@]}"; do
    read -ra mnk <<< "${shape}"
    m="${mnk[0]}"
    n="${mnk[1]}"
    k="${mnk[2]}"

    echo "GEMM shape (M, N, K) = (${m}, ${n}, ${k})"

    # The hipBLASLt equation is:
    #     D = activation( alpha ⋅ op(A) ⋅ op(B) + beta ⋅ op(C) + bias )
    # * op() refers to in-place operations, such as transpose and non-transpose
    # * alpha and beta are scalars
    # * activation function supports GELU and ReLU
    # * bias vector matches matrix D rows and broadcasts to all D columns.

    # The default values for the following parameters are:
    # * alpha = 1 and beta = 0
    # * bias = none
    # * activation = none
    # So, the equation becomes:
    #     D = op(A) ⋅ op(B)

    HIP_FORCE_DEV_KERNARG=1 hipblaslt-bench \
        --function matmul \
        -m "${m}" -n "${n}" -k "${k}" \
        --transA T --transB N \
        --precision i8_r --compute_type i32_r \
        --cold_iters 100 --iters 1000 \
        --algo_method all \
        --print_kernel_info \
        --flush --rotating 512 \
        &> "${output_dir}/${m}_${n}_${k}.txt"
done

echo 'Done.'
