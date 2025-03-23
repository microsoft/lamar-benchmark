#!/usr/bin/env bash

root_folder=$(realpath $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..)
source ${root_folder}/scripts/load_env.sh

apt-get install -y --no-install-recommends --no-install-suggests \
    cmake \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    build-essential

git clone -b 2.1.0 https://github.com/ceres-solver/ceres-solver.git ceres-solver-v2.1.0 --depth=1
cd ceres-solver-v2.1.0
cmake -S . -B build
cmake --build build --target install -- -j$(nproc)
