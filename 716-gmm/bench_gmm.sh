#!/usr/bin/env bash


script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)


function log() {
    timestamp=$(date +'%Y-%m-%d %H:%M:%S,%3N')
    echo "${timestamp} > ${*}"
}


function clean_triton_cache() {
    triton_cache_dir="${HOME}/.triton/cache"
    log 'Cleaning Triton cache to avoid filling up storage...'
    rm --recursive --force "${triton_cache_dir}"
}


function bench() {
    shape="${1}"
    read -r m k n g <<< "${shape}"
    shift

    clean_triton_cache

    python "${script_dir}/gmm.py" \
        "${m}" "${k}" "${n}" "${g}" \
        --bench --verbose --num-group-sizes 20 "${@}" 2>&1
}


function bench_layouts() {
    workload="${1}"

    shape="${2}"
    read -r m k n g <<< "${shape}"
    base_bench_log_file="${script_dir}/bench_${m}_${k}_${n}_${g}"

    log "Benchmarking shape (M, K, N, G) = (${m}, ${k}, ${n}, ${g}) for ${workload} workload..."

    # inference layouts:
    # * TN: row-major x column-major

    # training layouts:
    # * TN: row-major x column-major
    # * NN: column-major x column-major
    # * NT: column-major x row-major

    # TN: row-major x column-major => row-major
    log 'TN layout: inference + training'
    bench "${shape}" | tee "${base_bench_log_file}_rcr.log"

    if [ "${workload}" == 'training' ]; then
        # NN: column-major x column-major => row-major
        log 'NN layout: training'
        bench "${shape}" --trans-lhs | tee "${base_bench_log_file}_ccr.log"

        # NT: column-major x row-major => row-major
        log 'NT layout: training'
        bench "${shape}" --trans-lhs --no-trans-rhs | tee "${base_bench_log_file}_crr.log"
    fi
}


function main() {
    workload="${1}"

    log 'BIG BENCHMARK STARTED!'

    log 'Removing old benchmark log files...'
    rm --recursive --force "${script_dir}"/*.log

    export PYTHONWARNINGS='ignore'
    torch_version=$(pip show torch | grep Version | cut --delimiter ' ' --field 2)
    triton_version=$(pip show triton | grep Version | cut --delimiter ' ' --field 2)
    log "PyTorch version = ${torch_version}" | tee "${script_dir}/version.log"
    log "Triton version = ${triton_version}" | tee --append "${script_dir}/version.log"

    shapes=(
        '  49152  1408  2048 64'
        '3145728  2048  1408  8'
        ' 393216  2048  1408 64'
        '  32768  6144 16384  8'
        '  32768 16384  6144  8'
    )
    for shape in "${shapes[@]}"; do
        bench_layouts "${workload}" "${shape}"
    done

    grep best_config "${script_dir}/bench_"*.log \
        | tr --squeeze-repeats ' ' \
        | cut --delimiter ' ' --fields 6- \
        | sort \
        | uniq > "${script_dir}/best_configs.log"

    log 'BIG BENCHMARK ENDED!'
}


# main inference
main training
