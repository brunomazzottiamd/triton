#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${script_dir}/common.sh"

clean_artifacts softmax

shapes=(
    '4096, 8192'
    '8192, 8192'
)

echo 'Running Triton softmax...'
run_python softmax "${shapes[@]}" --save-tensors

echo 'Running Mojo softmax...'
run_mojo softmax "${shapes[@]}" --save-tensors > /dev/null

echo 'Running correctness test...'
run_test softmax

clean_tensors softmax
