#!/usr/bin/env bash

PS4='\033[0;32m$(date "+%Y%m%d %H:%M:%S.%N") $BASH_SOURCE:$LINENO]\033[0m '
set -euxo pipefail

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
