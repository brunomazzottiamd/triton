#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

N=30

for (( i=1; i<=N; i++ )); do
    rm --recursive --force ~/.triton/cache
    "${script_dir}/triton_tem_fused_addmm_130.py" bench
done | \
awk '
BEGIN {
    FS = "[[:space:]]+";
    baseline_sum = 0;
    optimized_sum = 0;
    baseline_sum_sq = 0;
    optimized_sum_sq = 0;
    count = 0;
    print "    Baseline Optimized";
}

# Performance line:
/^[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+$/ {
    baseline = $5;
    optimized = $6;
    baseline_sum += baseline;
    optimized_sum += optimized;
    baseline_sum_sq += baseline * baseline;
    optimized_sum_sq += optimized * optimized;
    count++;
    printf "%03d %.2f   %.2f\n", count, baseline, optimized;
}

# Best config line:
/^Best optimized tuning config: .+$/ {
    config = $0;
    sub(/Best optimized tuning config: /, "", config);
    sub(/, num_ctas: 1/, "", config);
    sub(/, maxnreg: None/, "", config);
    configs[config]++;
    # printf "Best config: %s\n", config
}

END {
    if (count > 0) {
        baseline_mean = baseline_sum / count;
        optimized_mean = optimized_sum / count;
        baseline_stdev = sqrt((baseline_sum_sq / count) - (baseline_mean * baseline_mean));
        optimized_stdev = sqrt((optimized_sum_sq / count) - (optimized_mean * optimized_mean));
        printf "\n Baseline: %.2f ± %.2f\n", baseline_mean, baseline_stdev;
        printf "Optimized: %.2f ± %.2f\n", optimized_mean, optimized_stdev;
	if (baseline_mean != 0) {
	   mean_speedup = optimized_mean / baseline_mean
	   printf "\nMean Speedup (Optimized / Baseline): %.2f\n", mean_speedup;
	}
	print "\nConfigs Frequency:"
	for (config in configs) {
	    printf "%6.2f%%: %s\n", 100 * configs[config] / count, config
	}
    } else {
        print "No data to process.";
    }
}
'
