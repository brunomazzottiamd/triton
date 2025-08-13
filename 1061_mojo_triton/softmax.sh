#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${script_dir}/common.sh"

clean_artifacts softmax

shapes=(
    ' 256,  128'
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

echo 'Profiling softmax...'
for shape in "${shapes[@]}"; do
    IFS=',' read -r m n <<< "${shape}"
    m="${m//[[:space:]]/}"
    n="${n//[[:space:]]/}"
    formatted_shape=$(printf '%05d_%05d' "${m}" "${n}")
    echo "Profiling Triton implementation for shape=(${m}, ${n})..."
    profile_python softmax softmax_kernel "softmax/${formatted_shape}" "${shape}"
    echo "Profiling Mojo implementation for shape=(${m}, ${n})..."
    profile_mojo softmax softmax_ref_softmax_kernel "softmax/${formatted_shape}" "${shape}"
done

run_python agg_profiling --kernel softmax
