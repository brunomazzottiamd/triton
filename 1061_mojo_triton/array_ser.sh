#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
triton_dir="${script_dir}/triton"
mojo_dir="${script_dir}/mojo"

python "${triton_dir}/array_ser.py"

pixi run --manifest-path="${mojo_dir}/pixi.toml" \
    mojo run "${mojo_dir}/array_ser.mojo"
