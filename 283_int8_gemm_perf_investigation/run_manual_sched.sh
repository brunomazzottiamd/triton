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
    output_dir="${4}"
    echo "Getting kernel ${kernel_file_desc}..."
    kernel_file=$(find "${triton_cache_dir}" -name "*.${kernel_file_ext}" | head -1)
    echo "Kernel ${kernel_file_desc} is [${kernel_file}]."
    cp "${kernel_file}" "${output_dir}"
}

TRITON_CACHE_DIR="${HOME}/.triton/cache"

clean_triton_cache() {
    echo "Cleaning Triton cache at [${TRITON_CACHE_DIR}]..."
    remove "${TRITON_CACHE_DIR}"
}


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


### Run Triton kernel without assembly injection

echo 'Running kernel to get reference assembly...'

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
"${kernel_program[@]}"

copy_kernel_file 'assembly' 'amdgcn' "${TRITON_CACHE_DIR}" "${output_dir}"


### Run Triton kernel with assembly injection

echo 'Running kernel with assembly injection...'

clean_triton_cache

export AMD_INSERT_AMDGCN="${script_dir}/my_asm.amdgcn"

pytest -vvv "${kernel_source}" \
    && python "${kernel_source}" bench \
    && diff --report-identical-files "${output_dir}"/*.amdgcn "${AMD_INSERT_AMDGCN}" > "${output_dir}/asm.diff"

unset AMD_INSERT_AMDGCN


### Done

echo 'Done.'
