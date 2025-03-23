#!/usr/bin/env bash

root_folder=$(realpath $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..)
source ${root_folder}/scripts/load_env.sh

# Uncomment the following line if you want to use this script inside Docker.
# apt-get update && apt-get install -y --no-install-recommends --no-install-suggests git python3 python3-dev python3-pip python-is-python3 sudo

# Create external folder.
mkdir ${root_folder}/external && cd ${root_folder}/external

# Ceres Solver.
sudo ${root_folder}/scripts/install_ceres_solver.sh

# Colmap.
sudo ${root_folder}/scripts/install_colmap.sh

# HLoc.
${root_folder}/scripts/install_hloc.sh
