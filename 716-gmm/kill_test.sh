#!/usr/bin/env bash

pid=$( \
    ps aux \
	| grep pytest \
	| grep --invert-match grep \
	| tr -s ' ' \
	| cut -d ' ' -f2 \
   )

echo "Test PID is ${pid}."

kill "${pid}"

rm --force "gpucore.${pid}"
