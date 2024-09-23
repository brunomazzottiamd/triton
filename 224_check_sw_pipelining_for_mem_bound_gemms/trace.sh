#!/usr/bin/env bash

### Helper functions

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

### Set output directory

echo "TRITON_HIP_USE_NEW_STREAM_PIPELINE = [${TRITON_HIP_USE_NEW_STREAM_PIPELINE}]"

if [ "${TRITON_HIP_USE_NEW_STREAM_PIPELINE}" != '1' ]; then
    # StreamPipelinerV1
    pipeliner='V1'
    num_stages=0
    output_dir="main_nS${num_stages}"
    echo "Using StreamPipelinerV1. num_stages is [${num_stages}]"
else
    # StreamPipelinerV2
    pipeliner='V2'
    num_stages="${1}"
    case "${num_stages}" in
        0|1|2|3|4)
            echo "num_stages [${num_stages}] is a valid one!"
            ;;
        *)
            echo "num_stages [${num_stages}] is invalid! Provide a number in [0, 4]. Exiting..."
            exit 1
            ;;
    esac
    output_dir="new_sp_nS${num_stages}"
    echo "Using StreamPipelinerV2. num_stages is [${num_stages}]"
fi

output_zip="$(basename "${output_dir}").zip"

echo "Output directory is [${output_dir}], it'll be compressed to [${output_zip}]."

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

kernel_program=(python "${script_dir}/test_matmul.py" -m 1 -n 6144 -k 6144)
if [ "${pipeliner}" == 'V2' ]; then
    kernel_program+=(--new_pipeliner --num_stages "${num_stages}")
fi
echo 'Kernel program is [' "${kernel_program[@]}" '].'

dispatch_id=$(rocprofv2 \
    "${kernel_program[@]}" \
    | grep 'matmul_kernel' \
    | cut --delimiter ',' --fields 1 \
    | sed 's/Dispatch_ID(//;s/)//'
)

echo "Kernel dispatch ID is ${dispatch_id}."

### Get kernel IRs and assembly code

copy_kernel_file 'Triton IR' 'ttir' "${triton_cache_dir}" "${output_dir}"
copy_kernel_file 'Triton GPU IR' 'ttgir' "${triton_cache_dir}" "${output_dir}"
copy_kernel_file 'assembly' 'amdgcn' "${triton_cache_dir}" "${output_dir}"

# Copy reference Python code too.
cp "${script_dir}/matmul_kernel.py" "${output_dir}"

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
