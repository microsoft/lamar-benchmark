#!/usr/bin/env bash

PS4='\033[0;32m$(date "+%Y%m%d %H:%M:%S.%N") $BASH_SOURCE:$LINENO]\033[0m '
set -euxo pipefail

git clone --recursive -b v1.4 https://github.com/cvg/Hierarchical-Localization/ hloc --depth=1
cd hloc
python -m pip install -e .
