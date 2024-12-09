#!/usr/bin/env bash


### Helper functions

remove() {
    rm --recursive --force "${@}"
}

clean_triton_cache() {
    triton_cache_dir="${HOME}/.triton/cache"
    echo "Cleaning Triton cache at [${triton_cache_dir}]..."
    remove "${triton_cache_dir}"
}


# ENTRY POINT

echo 'RUNNING TRITON KERNEL WITH MANUAL ASSEMBLY SCHEDULING...'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
kernel_source="${script_dir}/triton_tem_fused_addmm_130.py"
kernel_variation='opt'
kernel_name='triton_tem_fused_addmm_130_kernel_opt_no_autotune'
kernel_program=(python "${kernel_source}" "${kernel_variation}")


### Create new empty output directory

output_dir=$(mktemp --directory)

echo "Creating new empty output directory [${output_dir}]..."


### Run Triton kernel with manual schedule assembly injection

echo 'Running kernel with manual schedule assembly injection...'

export AMD_INSERT_AMDGCN_KERNEL="${kernel_name}"
export AMD_INSERT_AMDGCN_FILE="${script_dir}/${AMD_INSERT_AMDGCN_KERNEL}.amdgcn"

N=30

for (( i=1; i<=N; i++ )); do
    clean_triton_cache

    output_file="${output_dir}/prof_results_msched_${i}.csv"
    rocprof \
	--stats \
	-o "${output_file}" \
	"${kernel_program[@]}"

    stats_file="${output_dir}/prof_results_msched_${i}.stats.csv"
    kernel_time_ns=$(grep "${kernel_name}" "${stats_file}" | cut --delimiter ',' --fields 3)
    echo "kernel_time_ns,${kernel_time_ns}"
done | \
awk '
BEGIN {
    FS = ",";
    m = 84122;
    n = 2048;
    k = 256;
    tops = 2 * m * n * k * 1e-12;
    sum = 0;
    sum_sq = 0;
    count = 0;
}
/^kernel_time_ns,[0-9]+$/ {
    nanoseconds = $2;
    seconds = nanoseconds * 1e-9;
    tflops = tops / seconds;
    sum += tflops;
    sum_sq += tflops * tflops;
    count++;
    printf "%02d %.2f TFLOPS\n", count, tflops
}
{
    print
}
END {
    if (count > 0) {
        mean = sum / count;
        stdev = sqrt((sum_sq / count) - (mean * mean));
        printf "Mean: %.2f ± %.2f TFLOPS\n", mean, stdev;
    } else {
        print "No data to process.";
    }
}
'


### Cleanup intermediate files

echo 'Cleaning intermediate files...'

remove "${output_dir}"


### Done

echo 'Done.'
