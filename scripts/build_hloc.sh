#!/usr/bin/env bash

root_folder=$(realpath $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..)
source ${root_folder}/scripts/load_env.sh

# Clone hloc.
git clone --recursive -b v1.4 https://github.com/cvg/Hierarchical-Localization/ hloc --depth=1
cd hloc

# Build the wheel.
pip wheel --no-deps -w dist-wheel .
whl_path=$(find dist-wheel/ -name "*.whl")
echo $whl_path >dist-wheel/whl_path.txt
