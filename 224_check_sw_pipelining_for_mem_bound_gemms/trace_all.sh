#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

export TRITON_HIP_USE_NEW_STREAM_PIPELINE=0
"${script_dir}/trace.sh"

export TRITON_HIP_USE_NEW_STREAM_PIPELINE=1
for num_stages in $(seq 1 4); do
    "${script_dir}/trace.sh" "${num_stages}"
done
