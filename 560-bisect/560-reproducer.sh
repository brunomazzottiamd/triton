#!/usr/bin/env bash

export PIP_ROOT_USER_ACTION=ignore
export TRITON_BUILD_WITH_CCACHE=true

# Best performance:

echo 'Checkouting best performance commit...'
git switch --detach 1b0f9ea7f
echo 'Best performance commit is:'
git log -1 --pretty=format:"%cd | %h | %s" --date=short

echo 'Compiling Triton at best performance commit...'
{ pip uninstall --yes triton && \
  pushd 'python' || exit && \
  pip install --verbose --no-build-isolation . && \
  popd || exit; } &> best_commit_compilation.log

echo 'Checkouting kernel source commit (from ROCm main_perf)...'
git switch --detach eb7e0158f
echo 'Kernel source commit is:'
git log -1 --pretty=format:"%cd | %h | %s" --date=short

echo 'Running non-causal benchmark at best performance commit:'
pushd 'python/perf-kernels' &> /dev/null || exit && \
python flash-attention.py -b 1 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128 && \
popd &> /dev/null || exit

echo 'Running causal benchmark at best performance commit:'
pushd 'python/perf-kernels' &> /dev/null || exit && \
python flash-attention.py -b 1 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128 -causal && \
popd &> /dev/null || exit

# Worst performance:

echo 'Checkouting worst performance commit...'
git switch --detach 67dc6270a
echo 'Worst performance commit is:'
git log -1 --pretty=format:"%cd | %h | %s" --date=short

echo 'Compiling Triton at worst performance commit...'
{ pip uninstall --yes triton && \
  pushd 'python' || exit && \
  pip install --verbose --no-build-isolation . && \
  popd || exit; } &> worst_commit_compilation.log

echo 'Checkouting kernel source commit (from ROCm main_perf)...'
git switch --detach eb7e0158f
echo 'Kernel source commit is:'
git log -1 --pretty=format:"%cd | %h | %s" --date=short

echo 'Running non-causal benchmark at worst performance commit:'
pushd 'python/perf-kernels' &> /dev/null || exit && \
python flash-attention.py -b 1 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128 && \
popd &> /dev/null || exit

echo 'Running causal benchmark at worst performance commit:'
pushd 'python/perf-kernels' &> /dev/null || exit && \
python flash-attention.py -b 1 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128 -causal && \
popd &> /dev/null || exit

# The end:

echo 'DONE!'
echo 'Your Triton repository is in detached HEAD state, please take care.'
