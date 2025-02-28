#!/usr/bin/env bash


function prof_rmsnorm_bwd() {
    n="${1}"
    shift
    output_csv_file="${1}"
    shift
    echo 'kernel_name,duration_ns' > "${output_csv_file}"
    for ((i = 0; i < n; ++i)); do
	echo -n '.'
	rocprof --stats \
		python rmsnorm.py --no_benchmark -M 4096 -N 5120 --dtype bf16 --mode bwd "${@}" \
	        &> /dev/null
	grep 'bwd' results.stats.csv \
	    | cut --delimiter=',' --fields=1,3 \
	    >> "${output_csv_file}"
	rm --recursive --force results.*
    done
    echo
}


function compute_stats() {
    input_csv_file="${1}"
    python << EOF
import pandas as pd
us = pd.read_csv("${input_csv_file}")["duration_ns"] / 1000
trim_percent = 0.2  # Adjust the trimming percentage (20%)
trimmed_us = us.sort_values().iloc[int(len(us) * trim_percent) : int(len(us) * (1 - trim_percent))]
mean, std = trimmed_us.agg(["mean", "std"])
print(f"Trimmed sample size = {len(trimmed_us)}, kernel mean execution time = {mean:.2f} ± {std:.2f} µs")
EOF
}


N=25

echo 'Profiling RMSNorm backward pass implemented with 2 kernels...'
two_kernels_csv='2_kernels.csv'
prof_rmsnorm_bwd "${N}" "${two_kernels_csv}"
two_kernels_1st_csv='2_kernels_1st.csv'
grep --invert-match '_rmsnorm_bwd_dg_reduce' "${two_kernels_csv}" > "${two_kernels_1st_csv}"
echo '1st backward kernel statistics:'
compute_stats "${two_kernels_1st_csv}"
two_kernels_2nd_csv='2_kernels_2nd_reduce.csv'
grep --invert-match 'rms_bwd_kernel' "${two_kernels_csv}" > "${two_kernels_2nd_csv}"
echo '2nd reduction kernel statistics:'
compute_stats "${two_kernels_2nd_csv}"
rm --recursive --force "${two_kernels_csv}"

echo 'Profiling RMSNorm backward pass implemented with 1 kernel and atomic ops...'
one_kernel_csv='1_kernel+atomics.csv'
prof_rmsnorm_bwd "${N}" "${one_kernel_csv}" --dg_atomic
echo 'Backward kernel with atomics statistics:'
compute_stats "${one_kernel_csv}"

echo 'Done.'
