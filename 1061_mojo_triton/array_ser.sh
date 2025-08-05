#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${script_dir}/common.sh"

clean_tensors 'array_ser'

echo 'Running Triton array serialization...'
run_python 'array_ser'

echo 'Running Mojo array serialization...'
run_mojo 'array_ser'

echo 'Running Triton array serialization again...'
run_python 'array_ser'

clean_tensors 'array_ser'
