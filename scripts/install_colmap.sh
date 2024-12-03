#!/usr/bin/env bash

root_folder=$(realpath $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..)
source ${root_folder}/scripts/load_env.sh

apt-get install -y --no-install-recommends --no-install-suggests \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev

# Installing COLMAP version 3.8
git clone --recursive -b 3.8 https://github.com/colmap/colmap colmap_v3.8 --depth=1
cd colmap_v3.8
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=all
cmake --build build --target install -- -j$(nproc)
