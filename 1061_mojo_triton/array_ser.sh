#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
python_dir="${script_dir}/python"
mojo_dir="${script_dir}/mojo"
tensors_dir="${script_dir}/tensors"

rm --force --recursive "${tensors_dir}/"*array_ser*

echo 'Running Triton array serialization...'
python "${python_dir}/array_ser.py"

echo 'Running Mojo array serialization...'
pixi run --manifest-path="${mojo_dir}/pixi.toml" \
    mojo run "${mojo_dir}/array_ser.mojo"

echo 'Running Triton array serialization again...'
python "${python_dir}/array_ser.py"
