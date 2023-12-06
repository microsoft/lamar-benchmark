#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.

PS4='\033[1;96m$(date +%H:%M:%S)\033[0m '
set -exo pipefail

# Install Embree following the official instructions and set the environmental
# variable embree_DIR to point to embree-config.cmake. On Linux, this can be
# done as follows:
wget https://github.com/embree/embree/releases/download/v3.12.2/embree-3.12.2.x86_64.linux.tar.gz
tar xvzf embree-3.12.2.x86_64.linux.tar.gz
rm embree-3.12.2.x86_64.linux.tar.gz
mv embree-3.12.2.x86_64.linux embree-3.12.2
export embree_DIR=`readlink -f embree-3.12.2/lib/cmake/embree-3.12.2`

# Install RayBender.
pip install git+https://github.com/cvg/raybender.git -vv

# Clean up.
rm -fr embree-3.12.2
