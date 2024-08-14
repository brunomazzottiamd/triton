#!/usr/bin/env bash

### Helper functions

trim_string() {
    : "${1#"${1%%[![:space:]]*}"}"
    : "${_%"${_##*[![:space:]]}"}"
    printf '%s\n' "$_"
}

join_by() {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

### Create empty results directory

results_dir=$(trim_string "${1}")

if [ -z "${results_dir}" ]; then
    results_dir=$(date '+results_%Y-%m-%dT%H:%M:%S')
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
results_dir="${script_dir}/${results_dir}"

mkdir --parents "${results_dir}"

### Tune GEMM

echo 'Tuning...'

tune_gemm_py="${script_dir}/tune_gemm/tune_gemm.py"
tuning_results="${results_dir}/02_gemm_output.yaml"
gpu_ids=$(join_by , 0 1 2 3)
dtype='fp16'
dtypes=('-dtype_a' "${dtype}" '-dtype_b' "${dtype}" '-dtype_c' "${dtype}")

python "${tune_gemm_py}" \
    --gemm_size_file "${script_dir}/gemm_input.yaml" \
    --o "${tuning_results}" \
    --jobs 4 \
    --gpu_ids="${gpu_ids}" \
    "${dtypes[@]}" \
    | tee "${results_dir}/01_tune.txt"

### Check correctness

echo 'Checking correctness...'

python "${tune_gemm_py}" \
    --gemm_size_file "${tuning_results}" \
    --compare_wo_tuning \
    "${dtypes[@]}" \
    | tee "${results_dir}/03_check.txt"

### Benchmark

echo 'Benchmarking...'

bench_results="${results_dir}/05_bench_results_tflops.csv"

python "${tune_gemm_py}" \
    --gemm_size_file "${tuning_results}" \
    --o "${bench_results}" \
    --benchmark \
    --gpu_ids="${gpu_ids}" \
    "${dtypes[@]}" \
    | tee "${results_dir}/04_bench.txt"

python "${script_dir}/tflops_to_gibps.py" \
    --input "${bench_results}" \
    --output "${bench_results/tflops/gibps}"

### The end!

echo 'Done.'
