#!/usr/bin/env bash

pid=$(pgrep --full pytest)
echo "Test PID is ${pid}."
kill "${pid}"
rm --force "gpucore.${pid}"
