#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
python_dir="${script_dir}/python"
mojo_dir="${script_dir}/mojo"
tensors_dir="${script_dir}/tensors"

rm --force --recursive "${tensors_dir}/"*vector_add*

n=(
         4096
         8192
        16384
        32768
        65536
       131072
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
python "${python_dir}/vector_add.py" "${n[@]}" --save-out --verbose

echo 'Running Mojo vector add...'
pixi run --manifest-path="${mojo_dir}/pixi.toml" \
    mojo run "${mojo_dir}/vector_add.mojo" "${n[@]}" --save-out --verbose
