#!/usr/bin/env bash


# This script requires an installation of hipBLAS-common.
# It was done with the following commands:
#   cd /triton_dev || exit 1
#   git clone git@github.com:ROCm/hipBLAS-common.git
#   cd hipBLAS-common || exit 1
#   python3 ./rmake.py --install

# This script requires a hipBLASLt repository.
# It was done with the following commands:
#   cd /triton_dev || exit 1
#   git clone git@github.com:ROCm/hipBLASLt.git


# Compile hipBLASLt:

export GFX_COMPILATION_ARCH='gfx942:xnack-'

cd /triton_dev/hipBLASLt || exit 1

git clean -xdf

# install.sh flags:
# -i|--install      - install after build
# -d|--dependencies - install build dependencies
# -c|--clients      - build library clients too (combines with -i & -d)
#                     -c flag is important, it installs hipBLASLt clients that
#                     are used in benchmarking.
./install.sh \
    -idc \
    --architecture="${GFX_COMPILATION_ARCH}"

# It seems -i flag from install.sh already calls CMake and install everything!
# TODO: Investigate if it's possible to get the assembly code of a GEMM after
#       installing hipBLASLt in the system.
exit 0

cd build || exit 1

CPLUS_INCLUDE_PATH=/usr/lib/clang/17/include cmake \
  -DCMAKE_LIBRARY_PATH=/opt/rocm-6.2.0/llvm/lib \
  -DAMDGPU_TARGETS="${GFX_COMPILATION_ARCH}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DTensile_LOGIC= \
  -DTensile_CODE_OBJECT_VERSION=default \
  -DTensile_CPU_THREADS= \
  -DTensile_LIBRARY_FORMAT=msgpack \
  -DBUILD_CLIENTS_SAMPLES=ON \
  -DBUILD_CLIENTS_TESTS=ON \
  -DBUILD_CLIENTS_BENCHMARKS=ON \
  -DCPACK_SET_DESTDIR=OFF \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm \
  ..

CPLUS_INCLUDE_PATH=/usr/lib/clang/17/include cmake \
  --build . -- -j"$(nproc)"

cmake --build . -- package
