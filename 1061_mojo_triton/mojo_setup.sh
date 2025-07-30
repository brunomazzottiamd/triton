#!/usr/bin/env bash

if ! command -v pixi >/dev/null 2>&1; then
    # Install pixi.
    curl -fsSL https://pixi.sh/install.sh | sh
    export PATH="${HOME}/.pixi/bin:${PATH}"
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
mojo_dir="${script_dir}/mojo"

if [ ! -d "${mojo_dir}" ]; then
    # Initialize pixi project.
    pixi init "${mojo_dir}" \
	 --channel https://conda.modular.com/max-nightly/ \
	 --channel conda-forge
    pushd "${mojo_dir}"
    pixi add modular
fi
