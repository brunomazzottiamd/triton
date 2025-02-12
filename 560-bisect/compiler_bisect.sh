#!/usr/bin/env bash

# Compile Triton:
export PIP_ROOT_USER_ACTION=ignore
export TRITON_BUILD_WITH_CCACHE=true
pip uninstall --yes triton && \
pushd /triton_dev/triton/python || exit && \
pip install --verbose --no-build-isolation . && \
popd || exit

# Clean Triton cache:
rm --recursive --force ~/.triton/cache

# Show Git commit:
git log -1 --pretty=format:"%cd | %h | %s" --date=short

# Benchmark non-causal:
echo 'Non-causal:'
pushd /triton_dev/triton/perf-kernels || exit
n=3
for ((i=1; i <= n; i++)); do
    python flash-attention.py \
	   -b 1 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128
done | awk '
BEGIN {
    count = 0;
    sum = 0
}
/^[0-9]+([[:space:]]+[0-9]+\.[0-9]+)+$/ {
    sum += $7;
    count++
    printf "%02d %.2f\n", count, $7
}
END {
    if (count > 0)
        printf "Triton non-causal TFLOPS average: %.2f\n", sum / count;
    else
        print "No matching data found."
}
'
popd || exit

# Benchmark causal:
echo 'Causal:'
pushd /triton_dev/triton/perf-kernels || exit
n=3
for ((i=1; i <= n; i++)); do
    python flash-attention.py \
	   -b 1 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128 -causal
done | awk '
BEGIN {
    count = 0;
    sum = 0
}
/^[0-9]+([[:space:]]+[0-9]+\.[0-9]+)+$/ {
    sum += $7;
    count++
    printf "%02d %.2f\n", count, $7
}
END {
    if (count > 0)
        printf "Triton causal TFLOPS average: %.2f\n", sum / count;
    else
        print "No matching data found."
}
'
popd || exit


# Clean benchmark files:
rm --recursive --force ./*.csv ./*.png ./*.html
