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

    local kernel="${1}"
    shift

    clean_triton_cache

    python "${script_dir}/gmm.py" \
        "${m}" "${k}" "${n}" "${g}" --unif-group-sizes "${@}"

    local triton_cache_dir="${HOME}/.triton/cache"
    assembly_file=$(find "${triton_cache_dir}" -name '*.amdgcn' -print -quit)
    sgpr_spills=$(grep '.sgpr_spill_count' "${assembly_file}" | awk '{ print $2}')
    vgpr_spills=$(grep '.vgpr_spill_count' "${assembly_file}" | awk '{ print $2}')

    echo "${m},${k},${n},${g},${layout},${kernel},${sgpr_spills},${vgpr_spills}"
}


function reg_spills_layouts() {
    echo 'M,K,N,G,Layout,Kernel,SGPR Spills,VGPR Spills'

    mapfile -t shapes < <(get_target_shapes)

    for shape in "${shapes[@]}"; do
        shape=$(tr --complement --delete '0-9 ' <<< "${shape}")

        reg_spills "${shape}" NN GMM --gmm-type gmm
        reg_spills "${shape}" NT GMM --gmm-type gmm --trans-rhs
        reg_spills "${shape}" NN PTGMM --gmm-type ptgmm
        reg_spills "${shape}" TN PTGMM --gmm-type ptgmm --trans-lhs
        reg_spills "${shape}" NN NPTGMM --gmm-type tgmm
        reg_spills "${shape}" TN NPTGMM --gmm-type tgmm --trans-lhs
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
