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

cd /root/hipBLASLt/build/release/clients/staging || exit 1


### Set output directory

output_dir=$(trim_string "${1}")
if [ -z "${output_dir}" ]; then
    # "${1}" is empty, use a sensible default as output directory.
    output_dir=$(date '+results_%Y-%m-%dT%H:%M:%S')
fi

output_zip="$(basename "${output_dir}").zip"

echo "Output directory is [${output_dir}]. It'll be compressed to [${output_zip}]."


### Cleanup older files from previous runs

echo 'Cleaning older files from previous runs...'

remove "${output_dir}" "${output_zip}"


### Get kernel dispatch ID

echo 'Getting kernel dispatch ID...'

# HIP_FORCE_DEV_KERNARG=1 ./hipblaslt-bench \
#     --function matmul \
#     -m "${m}" -n "${n}" -k "${k}" \
#     --transA T --transB N \
#     --precision i8_r --compute_type i32_r \
#     --cold_iters 0 --iters 1 \
#     --api_method cpp --algo_method index --solution_index 67826
#     --print_kernel_info

# HIP_FORCE_DEV_KERNARG=1 ./hipblaslt-bench -f matmul -r f16_r -m 4864 -n 4096 -k 8192 --transA T \
# --transB N --compute_type f32_r --api_method cpp --algo_method index --solution_index 67826 -i 10 \
# -j 100 --print_kernel_info
