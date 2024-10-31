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

copy_kernel_file() {
    kernel_file_desc="${1}"
    kernel_file_ext="${2}"
    triton_cache_dir="${3}"
    output_file="${4}"
    echo "Getting kernel ${kernel_file_desc}..."
    kernel_file=$(find "${triton_cache_dir}" -name "*.${kernel_file_ext}" | head -1)
    echo "Kernel ${kernel_file_desc} is [${kernel_file}]."
    cp "${kernel_file}" "${output_file}"
}

TRITON_CACHE_DIR="${HOME}/.triton/cache"

clean_triton_cache() {
    echo "Cleaning Triton cache at [${TRITON_CACHE_DIR}]..."
    remove "${TRITON_CACHE_DIR}"
}

KERNEL_NAME='_triton_gemm_a8w8_kernel_no_autotune'

get_kernel_time_ns() {
    stats_file="${1}"
    grep "${KERNEL_NAME}" "${stats_file}" | cut --delimiter ',' --fields 3
}

get_kernel_time_us() {
    stats_file="${1}"
    echo "scale=2; ($(get_kernel_time_ns "${stats_file}") / 1000) + 0.05" | bc --mathlib
}


### ENTRY POINT

echo 'TESTING TRITON KERNEL WITH MANUAL SCHEDULING...'


### Set output directory

output_dir=$(trim_string "${1}")
if [ -z "${output_dir}" ]; then
    # "${1}" is empty, use a sensible default as output directory.
    output_dir=$(date '+results_%Y-%m-%dT%H-%M-%S')
fi

echo "Output directory is [${output_dir}]."


### Cleanup older files from previous runs

echo 'Cleaning older files from previous runs...'

remove "${output_dir}"


### Create new empty output directory

echo "Creating new empty output directory [${output_dir}]..."

mkdir --parents "${output_dir}"


### Run reference Triton kernel

echo 'Running reference kernel to get baseline assembly and baseline performance...'

clean_triton_cache

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
kernel_source="${script_dir}/test_int8_gemm.py"

M=20
N=1920
K=13312

kernel_program=(
    python "${kernel_source}" best -M "${M}" -N "${N}" -K "${K}"
)

unset AMD_INSERT_AMDGCN
rocprof --stats -o "${output_dir}/prof_results_ref.csv" "${kernel_program[@]}"

ref_time=$(get_kernel_time_us "${output_dir}/prof_results_ref.stats.csv")
echo "Reference time is ${ref_time} us."

asm_ref_file="${output_dir}/asm_ref.amdgcn"
copy_kernel_file 'assembly' 'amdgcn' "${TRITON_CACHE_DIR}" "${asm_ref_file}"

pytest "${kernel_source}"
python "${kernel_source}" bench


### Run Triton kernel with manual schedule assembly injection

echo 'Running kernel with manual schedule assembly injection...'

clean_triton_cache

export AMD_INSERT_AMDGCN="${script_dir}/my_asm.amdgcn"
rocprof --stats -o "${output_dir}/prof_results_msched.csv" "${kernel_program[@]}"

msched_time=$(get_kernel_time_us "${output_dir}/prof_results_msched.stats.csv")
echo "Manual schedule time is ${msched_time} us."

asm_msched_file="${output_dir}/asm_msched.amdgcn"
cp "${AMD_INSERT_AMDGCN}" "${asm_msched_file}"

pytest "${kernel_source}"
python "${kernel_source}" bench

unset AMD_INSERT_AMDGCN


### Report differences

echo 'Reporting code and performance differences...'

diff --report-identical-files "${asm_ref_file}" "${asm_msched_file}" > "${output_dir}/asm.diff"


### Done

echo 'Done.'
