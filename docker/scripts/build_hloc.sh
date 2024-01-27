#!/usr/bin/env bash

PS4='\033[1;96m$(date +%H:%M:%S)\033[0m '
set -exo pipefail

# Clone hloc.
git clone --recursive https://github.com/cvg/Hierarchical-Localization/ hloc --depth=1
cd hloc

# Build the wheel.
pip wheel --no-deps -w dist-wheel .
whl_path=$(find dist-wheel/ -name "*.whl")
echo $whl_path >dist-wheel/whl_path.txt
