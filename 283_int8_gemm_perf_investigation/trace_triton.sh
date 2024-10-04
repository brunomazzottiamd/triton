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


### Start tracing script

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
kernel_source="${script_dir}/test_int8_gemm.py"

M=20
N=1920
K=13312

echo "TRACING TRITON KERNEL [${kernel_source}] FOR (M, N, K) = (${M}, ${N}, ${K})"


### Set output directory

output_dir=$(trim_string "${1}")
if [ -z "${output_dir}" ]; then
    # "${1}" is empty, use a sensible default as output directory.
    output_dir=$(date '+results_%Y-%m-%dT%H-%M-%S')
fi

output_zip="$(basename "${output_dir}").tar.xz"

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

kernel_program=(
    python "${kernel_source}" run -M "${M}" -N "${N}" -K "${K}"
)

dispatch_id=$(rocprofv2 \
    "${kernel_program[@]}" \
    | grep '_triton_gemm_a8w8_kernel' \
    | cut --delimiter ',' --fields 1 \
    | sed 's/Dispatch_ID(//;s/)//'
)

echo "Kernel dispatch ID is ${dispatch_id}."


### Get kernel IRs and assembly code

copy_kernel_file 'Triton IR' 'ttir' "${triton_cache_dir}" "${output_dir}"
copy_kernel_file 'Triton GPU IR' 'ttgir' "${triton_cache_dir}" "${output_dir}"
copy_kernel_file 'assembly' 'amdgcn' "${triton_cache_dir}" "${output_dir}"

# Copy reference Python code too.
cp "${kernel_source}" "${output_dir}"


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

# Remove large files, keep only the parsed ATT.
remove "${output_dir}"/*.out "${output_dir}"/*.att "${output_dir}"/*.txt

### Compress output directory
# It's easier to transfer a single zip file!

echo "Compressing output directory to [${output_zip}]..."

tar \
    -cf "${output_zip}" \
    -I 'xz -9e' \
    "${output_dir}"


### Cleanup intermediate files

echo 'Cleaning intermediate files...'

remove "${input_file}" "${output_dir}"


### Done

echo 'Done.'
