#!/usr/bin/env bash


script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)


function clean_triton_cache() {
    local triton_cache_dir="${HOME}/.triton/cache"
    rm --recursive --force "${triton_cache_dir}"
}


function get_target_shapes() {
    PYTHONPATH="${script_dir}" python << EOF
from common import REAL_SHAPES
print("\n".join(" ".join(str(dim) for dim in shape) for shape in REAL_SHAPES))
EOF
}


function reg_spills() {
    local shape="${1}"
    read -r m k n g <<< "${shape}"
    shift

    local layout="${1}"
    shift

    clean_triton_cache

    python "${script_dir}/gmm.py" \
        "${m}" "${k}" "${n}" "${g}" --layout "${layout}" --unif-group-sizes "${@}"

    local triton_cache_dir="${HOME}/.triton/cache"
    assembly_file=$(find "${triton_cache_dir}" -name '*.amdgcn' -print -quit)
    sgpr_spills=$(grep '.sgpr_spill_count' "${assembly_file}" | awk '{ print $2}')
    vgpr_spills=$(grep '.vgpr_spill_count' "${assembly_file}" | awk '{ print $2}')

    echo "${m},${k},${n},${g},${layout},${sgpr_spills},${vgpr_spills}"
}


function reg_spills_layouts() {
    echo 'M,K,N,G,Layout,SGPR Spills,VGPR Spills'

    mapfile -t shapes < <(get_target_shapes)

    for shape in "${shapes[@]}"; do
        shape=$(tr --complement --delete '0-9 ' <<< "${shape}")

        reg_spills "${shape}" TN
        reg_spills "${shape}" NN
        reg_spills "${shape}" NT
    done
}


function csv_to_markdown() {
    python -c "
import sys
import pandas as pd
print(pd.read_csv(sys.stdin).to_markdown(index=False))
"
}


function main() {
    clean_triton_cache
    reg_spills_layouts | csv_to_markdown
}


main
