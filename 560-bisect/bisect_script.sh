#!/usr/bin/env bash

git log -1 --pretty=format:"%h | %s | %cd" --date=short

rm -rf ~/.triton/cache

n=5
for ((i=1; i <= n; i++)); do
    python python/perf-kernels/flash-attention.py \
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
        printf "Triton TFLOPS average: %.2f\n", sum / count;
    else
        print "No matching data found."
}
'
