#!/usr/bin/env bash


usage() {
    echo "Usage: ${0} {aiter,pallas}"
    echo 'error: a single argument with the kernel name is required'
    exit 1
}


if [ "${#}" -ne 1 ]; then
    usage
fi

kernel="${1}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# `prof_kernel.sh` can be found at
# [https://github.com/brunomazzottiamd/docker/blob/main/cscripts/prof_kernel.sh].

if [ "${kernel}" == aiter ]; then
    echo 'PROFILING AITER MHA KERNEL...'
    prof_kernel.sh \
	-r _attn_fwd.kd \
	-o "${script_dir}/aiter_mha_kernel_prof_data" \
	-- python "${script_dir}/run_mha_kernel.py" --kernel "${kernel}"

elif [ "${kernel}" == pallas ]; then
    echo 'PROFILING PALLAS MHA KERNEL...'
    prof_kernel.sh \
	-r mha_forward__1.kd \
	-o "${script_dir}/pallas_mha_kernel_prof_data" \
	-- python "${script_dir}/run_mha_kernel.py" --kernel "${kernel}"

else
    usage
fi
