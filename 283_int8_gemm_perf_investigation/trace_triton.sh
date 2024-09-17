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

echo "TRACING TRITON KERNEL FOR (M, N, K) = (${M}, ${N}, ${K})"


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


### Cleanup Triton cache

triton_cache_dir="${HOME}/.triton/cache"

echo "Cleaning Triton cache at [${triton_cache_dir}]..."

remove "${triton_cache_dir}"


### Create new empty output directory

echo "Creating new empty output directory [${output_dir}]..."

mkdir --parents "${output_dir}"


### Get kernel dispatch ID

echo 'Getting kernel dispatch ID...'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

kernel_program=(
    python "${script_dir}/test_int8_gemm.py" run -M "${M}" -N "${N}" -K "${K}"
)

dispatch_id=$(rocprofv2 \
    "${kernel_program[@]}" \
    | grep '_triton_gemm_a8w8_kernel' \
    | cut --delimiter ',' --fields 1 \
    | sed 's/Dispatch_ID(//;s/)//'
)

echo "Kernel dispatch ID is ${dispatch_id}."


### Create rocprofv2 input file

echo 'Creating rocprofv2 input file...'

input_file=$(mktemp --quiet)

cat << EOF >> "${input_file}"
att: TARGET_CU=0
SE_MASK=0xFFF
SIMD_SELECT=0xF
ISA_CAPTURE_MODE=2
DISPATCH=${dispatch_id}
EOF

echo 'rocprofv2 input file content is:'
cat "${input_file}"


### Generate kernel execution trace

echo 'Generating kernel execution trace...'

rocprofv2 \
    --input "${input_file}" \
    --plugin att auto \
    --mode file \
    --output-directory "${output_dir}" \
    "${kernel_program[@]}"

remove "${output_dir}"/*.out

### Compress output directory
# It's easier to transfer a single zip file!

echo "Compressing output directory to [${output_zip}]..."

zip \
    -q \
    -9 \
    -r \
    "${output_zip}" \
    "${output_dir}"


### Cleanup intermediate files

echo 'Cleaning intermediate files...'

remove "${input_file}" "${output_dir}"


### Done

echo 'Done.'
