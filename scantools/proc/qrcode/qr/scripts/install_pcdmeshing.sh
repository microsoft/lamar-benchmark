#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.

PS4='\033[1;96m$(date +%H:%M:%S)\033[0m '
set -exo pipefail

sudo apt-get install -y libboost-dev
pip install git+https://github.com/cvg/pcdmeshing.git -vv
