#!/usr/bin/env bash

### Helper functions

remove() {
    rm --recursive --force "${@}"
}

### Loop over dot product implementations

declare -a dots=(
    'tl_dot'
    'multreduce'
)

for dot in "${dots[@]}"; do
    echo "Tracing dot product implementation [${dot}]..."

    ### Cleanup older files from previous runs

    echo 'Cleaning older files from previous runs...'

    # TODO: Make `output_dir` a script argument.
    output_dir="output_${dot}"
    output_zip="$(basename "${output_dir}").zip"

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
        python "${script_dir}/test_matmul.py" single --dot "${dot}" -m 1 -n 4096 -k 4096
    )

    dispatch_id=$(rocprofv2 \
        "${kernel_program[@]}" \
        | grep 'matmul_kernel' \
        | cut --delimiter ',' --fields 1 \
        | sed 's/Dispatch_ID(//;s/)//'
    )

    echo "Kernel dispatch ID is ${dispatch_id}."

    ### Get kernel assembly code

    echo 'Getting kernel assembly code...'

    kernel_asm=$(find "${triton_cache_dir}" -name '*.amdgcn' | head -1)

    echo "Kernel assembly code is [${kernel_asm}]."

    cp "${kernel_asm}" "${output_dir}"
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
done

### Done

echo 'Done.'
