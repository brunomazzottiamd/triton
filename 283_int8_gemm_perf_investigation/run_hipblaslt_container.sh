#!/usr/bin/env bash

docker run \
    -it \
    --name "$(whoami)_hipblaslt" \
    --device /dev/kfd \
    --device /dev/dri \
    --network host \
    --mount "type=bind,source=${HOME},target=/hhome" \
    rocm/pytorch-private:rocm6.2_hipblaslt_scxiao
