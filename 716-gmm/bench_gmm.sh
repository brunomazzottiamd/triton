#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

function bench() {
    python "${script_dir}/gmm.py" --bench --verbose --num-group-sizes 20 "${@}" 2>&1
}

rm --recursive --force "${script_dir}"/*.log

# row-major x row-major
bench --no-trans-rhs | tee "${script_dir}/bench_rr.log"

# row-major x column-major
bench | tee "${script_dir}/bench_rc.log"

# column-major x row-major
bench --trans-lhs --no-trans-rhs | tee "${script_dir}/bench_cr.log"

# column-major x column-major
bench --trans-lhs | tee "${script_dir}/bench_cc.log"

grep best_config "${script_dir}/bench_"*.log \
    | tr --squeeze-repeats ' ' \
    | cut --delimiter ' ' --fields 6- \
    | sort \
    | uniq > "${script_dir}/best_configs.log"
