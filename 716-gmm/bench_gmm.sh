#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

function bench() {
    python "${script_dir}/gmm.py" --bench --verbose --num-group-sizes 20 "${@}" 2>&1
}

rm --recursive --force "${script_dir}"/*.log

# row-major x row-major => row-major
bench --no-trans-rhs \
    | tee "${script_dir}/bench_rrr.log"

# row-major x row-major => column-major
bench --no-trans-rhs --trans-out \
    | tee "${script_dir}/bench_rrc.log"

# row-major x column-major => row-major
bench \
    | tee "${script_dir}/bench_rcr.log"

# row-major x column-major => column-major
bench --trans-out \
    | tee "${script_dir}/bench_rcc.log"

# column-major x row-major => row-major
bench --trans-lhs --no-trans-rhs \
    | tee "${script_dir}/bench_crr.log"

# column-major x row-major => column-major
bench --trans-lhs --no-trans-rhs --trans-out \
    | tee "${script_dir}/bench_crc.log"

# column-major x column-major => row-major
bench --trans-lhs \
    | tee "${script_dir}/bench_ccr.log"

# column-major x column-major => column-major
bench --trans-lhs --trans-out \
    | tee "${script_dir}/bench_ccc.log"

grep best_config "${script_dir}/bench_"*.log \
    | tr --squeeze-repeats ' ' \
    | cut --delimiter ' ' --fields 6- \
    | sort \
    | uniq > "${script_dir}/best_configs.log"
