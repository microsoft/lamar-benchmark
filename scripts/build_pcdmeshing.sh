#!/usr/bin/env bash

root_folder=$(realpath $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..)
source ${root_folder}/scripts/load_env.sh

sudo apt-get install -y --no-install-recommends --no-install-suggests \
         libboost-dev libgmp3-dev libmpfrc++-dev
git clone --recursive https://github.com/cvg/pcdmeshing.git --depth=1
cd pcdmeshing

# Build the wheel.
pip wheel --no-deps -w dist-wheel .
whl_path=$(find dist-wheel/ -name "*.whl")
echo $whl_path >dist-wheel/whl_path.txt
