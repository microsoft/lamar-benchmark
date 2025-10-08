#!/usr/bin/env bash

PS4='\033[0;32m$(date "+%Y%m%d %H:%M:%S.%N") $BASH_SOURCE:$LINENO]\033[0m '
set -euxo pipefail

sudo apt-get install -y --no-install-recommends --no-install-suggests \
         libboost-dev libgmp3-dev libmpfrc++-dev
git clone --recursive https://github.com/cvg/pcdmeshing.git --depth=1
cd pcdmeshing

cd pybind11
git fetch --tags
git checkout v2.13.6
cd ..

# Build the wheel.
pip wheel --no-deps -w dist-wheel .
whl_path=$(find dist-wheel/ -name "*.whl")
echo $whl_path >dist-wheel/whl_path.txt
