#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

python "${script_dir}/triton/array_ser.py"

mojo_dir="${script_dir}/mojo"
pixi run --manifest-path="${mojo_dir}/pixi.toml" \
     mojo run "${mojo_dir}/array_ser.mojo"
