#!/usr/bin/env bash


### Helper functions

trim_string() {
    : "${1#"${1%%[![:space:]]*}"}"
    : "${_%"${_##*[![:space:]]}"}"
    printf '%s\n' "$_"
}

remove() {
    rm --recursive --force "${@}"
}


### Start tracing script

M=20
N=1920
K=13312

echo "TRACING HIPBLASLT KERNEL FOR (M, N, K) = (${M}, ${N}, ${K})"


### Set output directory

output_dir=$(trim_string "${1}")
if [ -z "${output_dir}" ]; then
    # "${1}" is empty, use a sensible default as output directory.
    output_dir=$(date '+results_%Y-%m-%dT%H-%M-%S')
fi

output_zip="$(basename "${output_dir}").tar.zx"

echo "Output directory is [${output_dir}]. It'll be compressed to [${output_zip}]."


### Cleanup older files from previous runs

echo 'Cleaning older files from previous runs...'

remove "${output_dir}" "${output_zip}"


### Get kernel dispatch ID

echo 'Getting kernel dispatch ID...'

# Solution index can be found in "hipblaslt_bench_results/${M}_${N}_${K}_winner.txt" file.
solution_index=99867
# solution_index=613
HIP_FORCE_DEV_KERNARG=1 hipblaslt-bench \
    --function matmul \
    -m "${M}" -n "${N}" -k "${K}" \
    --transA T --transB N \
    --precision i8_r --compute_type i32_r \
    --cold_iters 0 --iters 1 \
    --api_method cpp --algo_method index --solution_index "${solution_index}" \
    --print_kernel_info

# Getting this error:
# error: NO solution found!

exit 0
