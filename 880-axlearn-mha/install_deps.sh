#!/usr/bin/env bash


# Important: run this script in the container.

# The container already has the following software:
# * Ubuntu 22.04.5 LTS
# * ROCm 6.4.0
# * Python 3.10.12
# * JAX 0.5.0


export PIP_ROOT_USER_ACTION=ignore


# Install AXlearn:
#-----------------------------------------------------------------------

cd /workspace || exit 1
git clone git@github.com:ROCm/axlearn.git

cd /workspace/axlearn || exit 1
git checkout pallas_v2
pip install --editable '.[core]'

pip show axlearn


# Install PyTorch
#-----------------------------------------------------------------------
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#using-wheels-package

pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/rocm6.4/

pip show torch


# Install Triton
#-----------------------------------------------------------------------

cd /workspace || exit 1
git clone git@github.com:triton-lang/triton.git

cd /workspace/triton || exit 1
pip install --requirement python/requirements.txt
pip install --verbose --no-build-isolation --editable .

pip show triton


# Install AITER
#-----------------------------------------------------------------------

cd /workspace || exit 1
git clone --recursive https://github.com/ROCm/aiter.git

cd /workspace/aiter || exit 1
python setup.py develop

pip show aiter


# Minor amenities
#-----------------------------------------------------------------------

apt-get --yes update
apt-get --yes upgrade
apt-get --yes install --no-install-recommends \
    less      \
    tree      \
    bc        \
    zip       \
    htop      \
    shellcheck
apt-get --yes autoremove
apt-get clean

pip install \
    ruff    \
    mypy    \
    ipython \
    csvkit
