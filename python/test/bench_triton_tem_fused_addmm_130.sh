#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

N=30

for (( i=1; i<=N; i++ )); do
    "${script_dir}/triton_tem_fused_addmm_130.py"
done | \
awk '
BEGIN {
    FS = "[[:space:]]+";
    baseline_sum = 0;
    optimized_sum = 0;
    baseline_sum_sq = 0;
    optimized_sum_sq = 0;
    count = 0;
    print "      Baseline  Optimized";
}

/^[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+$/ {
    baseline = $5;
    optimized = $6;
    baseline_sum += baseline;
    optimized_sum += optimized;
    baseline_sum_sq += baseline * baseline;
    optimized_sum_sq += optimized * optimized;
    count++;
    printf "%03d %.6f %.6f\n", count, baseline, optimized;
}

END {
    if (count > 0) {
        baseline_mean = baseline_sum / count;
        optimized_mean = optimized_sum / count;
        baseline_stdev = sqrt((baseline_sum_sq / count) - (baseline_mean * baseline_mean));
        optimized_stdev = sqrt((optimized_sum_sq / count) - (optimized_mean * optimized_mean));
        printf "\n Baseline: %.6f ± %.6f\n", baseline_mean, baseline_stdev;
        printf "Optimized: %.6f ± %.6f\n", optimized_mean, optimized_stdev;
	if (baseline_mean != 0) {
	   mean_speedup = optimized_mean / baseline_mean
	   printf "\nMean Speedup (Optimized / Baseline): %.6f\n", mean_speedup;
	}
    } else {
        print "No data to process.";
    }
}
'
