#!/usr/bin/env bash

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

no_compilation=false
while getopts "n" opt; do
    case "${opt}" in
	n)
	    no_compilation=true
	    ;;
	*)
	    echo "Usage: ${0} [-n]" >&2
	    exit 1
	    ;;
    esac
done

if [ "${no_compilation}" = false ]; then
    echo "Compiling Triton..."
    export PIP_ROOT_USER_ACTION=ignore
    export TRITON_BUILD_WITH_CCACHE=true
    pip uninstall --yes triton && \
    pushd "${script_dir}/../python" || exit && \
    pip install --verbose --no-build-isolation . && \
    popd || exit
else
    echo "Skipping Triton compilation."
fi

# Clean Triton cache:
rm --recursive --force ~/.triton/cache

# Show Git commit:
echo 'Current branch:'
git log -1 --pretty=format:"%cd | %h | %s" --date=short
echo 'main branch:'
git log main -1 --pretty=format:"%cd | %h | %s" --date=short

n=3
pushd "${script_dir}/../python/perf-kernels" || exit

# Benchmark non-causal:
echo 'Non-causal benchmark:'
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

# Benchmark causal:
echo 'Causal benchmark:'
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

# Clean benchmark files:
rm --recursive --force ./*.csv ./*.png ./*.html

popd || exit
