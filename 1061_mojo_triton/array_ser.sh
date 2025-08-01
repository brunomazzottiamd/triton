#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
python_dir="${script_dir}/python"
mojo_dir="${script_dir}/mojo"

python "${python_dir}/array_ser.py"

pixi run --manifest-path="${mojo_dir}/pixi.toml" \
    mojo run "${mojo_dir}/array_ser.mojo"
