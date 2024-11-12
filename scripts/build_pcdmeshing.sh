#!/usr/bin/env bash

PS4='\033[1;96m$(date +%H:%M:%S)\033[0m '
set -exo pipefail

sudo apt-get install -y --no-install-recommends --no-install-suggests \
         libboost-dev libgmp3-dev libmpfrc++-dev
git clone --recursive https://github.com/cvg/pcdmeshing.git --depth=1
cd pcdmeshing

# Build the wheel.
pip wheel --no-deps -w dist-wheel .
whl_path=$(find dist-wheel/ -name "*.whl")
echo $whl_path >dist-wheel/whl_path.txt
