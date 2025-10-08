#!/usr/bin/env bash

PS4='\033[0;32m$(date "+%Y%m%d %H:%M:%S.%N") $BASH_SOURCE:$LINENO]\033[0m '
set -euxo pipefail

# Clone raybender.
git clone --recursive https://github.com/cvg/raybender.git --depth=1
cd raybender

cd pybind11
git pull
cd ..

# Install Embree following the official instructions and set the environmental
# variable embree_DIR to point to embree-config.cmake. On Linux, this can be
# done as follows:
wget https://github.com/embree/embree/releases/download/v3.12.2/embree-3.12.2.x86_64.linux.tar.gz
tar xvzf embree-3.12.2.x86_64.linux.tar.gz
rm embree-3.12.2.x86_64.linux.tar.gz
mv embree-3.12.2.x86_64.linux embree-3.12.2
export embree_DIR=`readlink -f embree-3.12.2/lib/cmake/embree-3.12.2`

# Build the wheel.
pip wheel --no-deps -w dist-wheel .
whl_path=$(find dist-wheel/ -name "*.whl")
echo $whl_path >dist-wheel/whl_path.txt
