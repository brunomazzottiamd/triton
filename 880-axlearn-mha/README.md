# Scripts for "Investigate AXLearn FA Pallas kernel performance issues on MI300" GitHub issue

**Author:** Bruno Mazzotti | [bruno.mazzotti@amd.com](mailto:bruno.mazzotti@amd.com) | [bruno.mazzotti@dxc.com](mailto:bruno.mazzotti@dxc.com)

**Date:** June 24th, 2025

This directory contains some scripts being used in the scope of [Investigate AXLearn FA Pallas kernel performance issues on MI300](https://github.com/ROCm/triton-internal/issues/880) GitHub issue. Please check the GitHub issue to understand its context, goals and current state.

Next, each script file has its purpose explained.

## `run_docker.sh`

It downloads the base Docker image used by AXLearn team and spins up a development container. It should be executed in a MI300 host. The base Docker image contains Ubuntu 22.04.5 LTS, Python 3.10.12, ROCm 6.4.0 and JAX 0.5.0.

## `install_deps.sh`

It installs other dependencies on top of the base Docker image downloaded by `run_docker.sh`. It should be executed in the development container. Theoretically speaking, you should run this script just once. It installs AXLearn (from `pallas_v2` ROCm development branch), PyTorch (from PyPI), Triton (from upstream `main` branch, compiled from source) and AITER (from ROCm `main` branch).

The scripts described by this document are hosted at [Bruno's Triton fork](https://github.com/brunomazzottiamd/triton/tree/880-axlearn-mha/880-axlearn-mha) but `install_deps.sh` installs Triton compiler from upstream `main` branch.

## `run_mha_kernel.py`

Python script with a simple command line interface to execute MHA kernels of interest. It can run MHA kernels from AITER (good performance) and AXLearn (bad performance, trying to improve it).

We're interested in causal MHA with `fp16` data type. So, the command line interface just allows changing the shape of attention tensors.

This Python script can also compare the output of distinct MHA kernels. As can be seen on the [GitHub issue](https://github.com/ROCm/triton-internal/issues/880#issuecomment-2992313625), the outputs are very different.

## `prof_mha_kernel.sh`

This is a kernel profiling script. It executes `run_mha_kernel.py` with `rocprof` (V1) several times and computes execution time statistics. It relies on `prof_kernel.sh` script written by Bruno a long time ago, you can find `prof_kernel.sh` at [Bruno's Dockerfile repository](https://github.com/brunomazzottiamd/docker/blob/main/cscripts/prof_kernel.sh).

## `check_src.sh`

Runs code formatter and linter. It can be safely ignored for functional purposes.
