#!/usr/bin/env bash

PS4='\033[1;96m$(date +%H:%M:%S)\033[0m '
set -exo pipefail

docker build --target builder -t lamar:builder -f Dockerfile .
docker build --target scantools -t lamar:scantools -f Dockerfile .
docker build --target pyceres -t lamar:pyceres -f Dockerfile .
