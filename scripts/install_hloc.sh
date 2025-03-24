#!/usr/bin/env bash

root_folder=$(realpath $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..)
source ${root_folder}/scripts/load_env.sh

git clone --recursive -b v1.4 https://github.com/cvg/Hierarchical-Localization/ hloc --depth=1
cd hloc
python -m pip install -e .
