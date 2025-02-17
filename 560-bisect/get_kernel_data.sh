#!/usr/bin/env bash

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

pushd "${script_dir}/../python/perf-kernels" || exit

echo 'Non-causal kernel:'
prof_kernel.sh -n attn_fwd -o "${1}-non-causal" -- \
    python flash-attention.py -no_bench -b 1 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128

echo 'Causal kernel:'
prof_kernel.sh -n attn_fwd -o "${1}-causal" -- \
    python flash-attention.py -no_bench -b 1 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128 -causal

popd || exit
