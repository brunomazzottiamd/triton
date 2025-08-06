#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${script_dir}/common.sh"

clean_artifacts vector_add

ns=(
         4096
         5555
         8192
        16384
        32768
        65536
       131072
       222222
       262144
       524288
      1048576
      2097152
      4194304
      8388608
     16777216
     33554432
     67108864
    134217728
)

echo 'Running Triton vector add...'
run_python vector_add "${ns[@]}" --save-tensors

echo 'Running Mojo vector add...'
run_mojo vector_add "${ns[@]}" --save-tensors > /dev/null

echo 'Running correctness test...'
run_test vector_add

clean_tensors vector_add

echo 'Profiling vector add...'
for n in "${ns[@]}"; do
    formatted_n=$(printf '%09d' "${n}")
    echo "Profiling Triton implementation for n=${n}..."
    profile_python vector_add vector_add_kernel "vector_add/${formatted_n}" "${n}"
    echo "Profiling Mojo implementation for n=${n}..."
    profile_mojo vector_add vector_add_kernel "vector_add/${formatted_n}" "${n}"
done

run_python agg_profiling --kernel vector_add
