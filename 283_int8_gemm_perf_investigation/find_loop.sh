#!/usr/bin/env bash

for dir in $(find . -type d -name 'trace_manual_sched_1_adv_counters*'); do
    python find_loop.py "${dir}/_triton_gemm_a8w8_kernel_no_autotune.kd_v0_ui"
done
