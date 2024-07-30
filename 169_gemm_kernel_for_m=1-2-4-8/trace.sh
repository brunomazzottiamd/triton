#!/usr/bin/env bash

### Helper functions

remove() {
    rm --recursive --force "${@}"
}

### Cleanup Triton cache

echo 'Cleaning Triton cache...'

triton_cache_dir="${HOME}/.triton/cache"

remove "${triton_cache_dir}"

### Get kernel dispatch ID

echo 'Getting kernel dispatch ID...'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

kernel_program=(
    python "${script_dir}/test_matmul.py" single -m 1 -n 4096 -k 4096
)

dispatch_id=$(rocprofv2 \
    "${kernel_program[@]}" \
    | grep 'matmul_kernel' \
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

### Cleanup older files from previous runs

echo 'Cleaning older files from previous runs...'

# TODO: Make `output_dir` a script argument.
output_dir='output'
output_zip="$(basename ${output_dir}).zip"

remove "${output_dir}" "${output_zip}"

### Generate kernel execution trace

echo 'Generating kernel execution trace...'

rocprofv2 \
    --input "${input_file}" \
    --plugin att auto \
    --mode file \
    --output-directory "${output_dir}" \
    "${kernel_program[@]}"

### Compress output directory
# It's easier to transfer a single zip file!

echo 'Compressing output directory...'

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
