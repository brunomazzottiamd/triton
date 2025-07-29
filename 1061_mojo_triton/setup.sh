#!/usr/bin/env bash

if ! command -v pixi >/dev/null 2>&1; then
    # Install pixi.
    curl -fsSL https://pixi.sh/install.sh | sh
    export PATH="${HOME}/.pixi/bin:${PATH}"
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
project_dir="${script_dir}/mojo_triton"

if [ ! -d "${project_dir}" ]; then
    # Initialize pixi project.
    pixi init "${project_dir}" \
	 --channel https://conda.modular.com/max-nightly/ \
	 --channel conda-forge
    pushd "${project_dir}"
    # pixi add python=3.12 pytorch=2.7.1
    # pixi add modular
    pixi add mojo
fi
