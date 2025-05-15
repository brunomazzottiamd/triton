#!/usr/bin/env bash


script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)


function log() {
    timestamp=$(date +'%Y-%m-%d %H:%M:%S,%3N')
    echo "${timestamp} > ${*}"
}


function bench() {
    python "${script_dir}/gmm.py" --bench --verbose --num-group-sizes 20 "${@}" 2>&1
}


function main() {
    workload='inference'
    # workload='training'
    log "Benchmarking for ${workload} workload..."

    log 'Removing old benchmark log files...'
    rm --recursive --force "${script_dir}"/*.log

    export PYTHONWARNINGS='ignore'
    torch_version=$(pip show torch | grep Version | cut --delimiter ' ' --field 2)
    triton_version=$(pip show triton | grep Version | cut --delimiter ' ' --field 2)
    log "PyTorch version = ${torch_version}" | tee "${script_dir}/version.log"
    log "Triton version = ${triton_version}" | tee --append "${script_dir}/version.log"

    # inference layouts:
    # * TN: row-major x column-major

    # training layouts:
    # * TN: row-major x column-major
    # * NN: column-major x column-major
    # * NT: column-major x row-major

    # TN: row-major x column-major => row-major
    bench \
	| tee "${script_dir}/bench_rcr.log"

    if [ "${workload}" == 'training' ]; then
	# NN: column-major x column-major => row-major
	bench --trans-lhs \
	    | tee "${script_dir}/bench_ccr.log"

	# NT: column-major x row-major => row-major
	bench --trans-lhs --no-trans-rhs \
	    | tee "${script_dir}/bench_crr.log"
    fi

    grep best_config "${script_dir}/bench_"*.log \
	| tr --squeeze-repeats ' ' \
	| cut --delimiter ' ' --fields 6- \
	| sort \
	| uniq > "${script_dir}/best_configs.log"
}


main
