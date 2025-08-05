#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${script_dir}/common.sh"

clean_tensors 'vector_add'

n=(
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
run_python 'vector_add' "${n[@]}" --save-tensors

echo 'Running Mojo vector add...'
run_mojo 'vector_add' "${n[@]}" --save-tensors > /dev/null

echo 'Running correctness test...'
run_test 'vector_add'

clean_tensors 'vector_add'
