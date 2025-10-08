#!/usr/bin/env bash

PS4='\033[0;32m$(date "+%Y%m%d %H:%M:%S.%N") $BASH_SOURCE:$LINENO]\033[0m '
set -euxo pipefail

apt-get install -y --no-install-recommends --no-install-suggests \
    cmake \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    build-essential

git clone -b 2.1.0 https://github.com/ceres-solver/ceres-solver.git ceres-solver-v2.1.0 --depth=1
cd ceres-solver-v2.1.0
cmake -S . -B build
cmake --build build --target install -- -j$(nproc)
