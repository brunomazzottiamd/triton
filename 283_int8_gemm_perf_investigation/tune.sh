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

get_gpu_ids() {
    local ids
    case "$(hostname)" in
        'sc-am11-smc-01')
            ids=(0 1 2 3)
            ;;
        'smc300x-ccs-aus-GPUF292'|'smc300x-ccs-aus-GPUF2AB')
            ids=(5 6 7)
            ;;
	'banff-cyxtera-s74-1')
	    ids=(4 5 6 7)
	    ;;
        *)
            ids=(0)
            ;;
    esac
    join_by , "${ids[@]}"
}

### Create empty results directory

results_dir=$(trim_string "${1}")

if [ -z "${results_dir}" ]; then
    results_dir=$(date '+results_%Y-%m-%dT%H-%M-%S')
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
results_dir="${script_dir}/${results_dir}"

mkdir --parents "${results_dir}"

### Tune GEMM

gpu_ids=$(get_gpu_ids)
echo "Tuning with GPUs [${gpu_ids}]..."

tune_gemm_py="${script_dir}/tune_gemm/tune_gemm.py"
tuning_results="${results_dir}/02_gemm_output.yaml"
dtypes=('-dtype_a' 'int8' '-dtype_b' 'int8' '-dtype_c' 'fp16')

python "${tune_gemm_py}" \
    --gemm_size_file "${script_dir}/gemm_input.yaml" \
    --o "${tuning_results}" \
    --jobs 4 \
    --gpu_ids="${gpu_ids}" \
    --icache_flush \
    --rotating_tensor 512 \
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
    --icache_flush \
    --rotating_tensor 512 \
    "${dtypes[@]}" \
    | tee "${results_dir}/04_bench.txt"

tflops_to_gibps.py \
    --input "${bench_results}" \
    --output "${results_dir}/06_bench_results_gibps.csv" \
    --bytes-per-a-elem 1 --bytes-per-b-elem 1 --bytes-per-c-elem 2

### The end!

echo 'Done.'