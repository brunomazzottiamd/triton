#!/usr/bin/env bash

# This pgrep pattern matches runner, benchmark and unit test.
pid=$(pgrep --full gmm.py)
echo "Test PID is ${pid}."
kill "${pid}"
rm --force "gpucore.${pid}"
