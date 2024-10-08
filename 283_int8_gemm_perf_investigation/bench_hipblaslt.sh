#!/usr/bin/env bash

echo 'Running hipBLASLt benchmark...'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
output_dir="${script_dir}/hipblaslt_bench_results"
echo "Output directory is [${output_dir}]."

rm --recursive --force "${output_dir}"
mkdir --parents "${output_dir}"

declare -a target_shapes=(
    '20 1920 13312'
    '30 1920 13312'
    '20 17792 13312'
    '30 17792 13312'
)

output_winners_file="${output_dir}/winners.csv"
echo 'transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,rotating_buffer,hipblaslt-Gflops,hipblaslt-GB/s,us' > "${output_winners_file}"

for shape in "${target_shapes[@]}"; do
    read -ra mnk <<< "${shape}"
    m="${mnk[0]}"
    n="${mnk[1]}"
    k="${mnk[2]}"

    echo "Benchmarking GEMM shape (M, N, K) = (${m}, ${n}, ${k})..."

    output_file="${output_dir}/${m}_${n}_${k}.txt"
    output_winner_file="${output_dir}/${m}_${n}_${k}_winner.txt"

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
        &> "${output_file}"

    # Get winner:
    grep \
	--ignore-case \
	--after-context=5 \
	Winner \
	"${output_file}" \
	> "${output_winner_file}"

    # Append to winners file:
    sed \
	--quiet \
	3p \
	"${output_winner_file}" \
	| tr --delete ' ' \
        >> "${output_winners_file}"

    # Compress big output file:
    xz -9e "${output_file}"
done

echo 'Done.'
