#!/usr/bin/env bash


script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)


function log() {
    local timestamp
    timestamp=$(date +'%Y-%m-%d %H:%M:%S,%3N')
    echo "${timestamp} > ${*}"
}


function clean_triton_cache() {
    local triton_cache_dir="${1:-${HOME}/.triton/cache}"
    log "Cleaning Triton cache at [${triton_cache_dir}] to avoid filling up storage..."
    rm --recursive --force "${triton_cache_dir}"
}


function get_target_shapes() {
    PYTHONPATH="${script_dir}" python << EOF
from common import REAL_SHAPES
print("\n".join(" ".join(str(dim) for dim in shape) for shape in REAL_SHAPES))
EOF
}


function bench() {
    local shape="${1}"
    read -r m k n g <<< "${shape}"
    shift

    local triton_cache_dir="${1}"
    shift

    TRITON_CACHE_DIR="${triton_cache_dir}" python "${script_dir}/gmm.py" \
        "${m}" "${k}" "${n}" "${g}" \
        --bench --verbose --num-group-sizes 20 "${@}" 2>&1

    clean_triton_cache "${triton_cache_dir}"
}


function bench_layouts() {
    local workload="${1}"

    local shape="${2}"
    read -r m k n g <<< "${shape}"

    local base_bench_file="${script_dir}/bench_${m}_${k}_${n}_${g}"

    log "Benchmarking shape (M, K, N, G) = (${m}, ${k}, ${n}, ${g}) for ${workload} workload..."

    # inference layouts:
    # * TN: row-major x column-major

    # training layouts:
    # * TN: row-major x column-major
    # * NN: column-major x column-major
    # * NT: column-major x row-major

    # TN: row-major x column-major => row-major
    log 'TN layout: inference + training'
    local base_layout_file="${base_bench_file}_rcr"
    bench "${shape}" "${base_layout_file}_cache" | tee "${base_layout_file}.log"

    if [ "${workload}" == 'training' ]; then
        # NN: column-major x column-major => row-major
        log 'NN layout: training'
        local base_layout_file="${base_bench_file}_ccr"
        bench "${shape}" "${base_layout_file}_cache" --trans-lhs | tee "${base_layout_file}.log"

        # NT: column-major x row-major => row-major
        log 'NT layout: training'
        local base_layout_file="${base_bench_file}_crr"
        bench "${shape}" "${base_layout_file}_cache" --trans-lhs --no-trans-rhs | tee "${base_layout_file}.log"
    fi
}


function main() {
    workload="${1}"

    log 'BIG BENCHMARK STARTED!'

    clean_triton_cache

    log 'Removing old benchmark log files...'
    rm --recursive --force "${script_dir}"/*.log

    export PYTHONWARNINGS='ignore'
    torch_version=$(pip show torch | grep Version | cut --delimiter ' ' --field 2)
    triton_version=$(pip show triton | grep Version | cut --delimiter ' ' --field 2)
    log "PyTorch version = ${torch_version}" | tee "${script_dir}/version.log"
    log "Triton version = ${triton_version}" | tee --append "${script_dir}/version.log"

    mapfile -t shapes < <(get_target_shapes)

    for shape in "${shapes[@]}"; do
        shape=$(tr --complement --delete '0-9 ' <<< "${shape}")
        bench_layouts "${workload}" "${shape}"
    done

    grep best_config "${script_dir}/bench_"*.log \
        | tr --squeeze-repeats ' ' \
        | cut --delimiter ' ' --fields 6- \
        | sort \
        | uniq > "${script_dir}/best_configs.log"

    log 'Compressing log files...'
    pushd "${script_dir}" &> /dev/null || return
    zip -q -9 bench_gmm_logs.zip ./*.log
    popd &> /dev/null || return

    log 'BIG BENCHMARK ENDED!'
}


# main inference
main training
