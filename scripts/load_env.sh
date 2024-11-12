#!/usr/bin/env bash

# PS4='\033[1;96m$(date +%H:%M:%S)\033[0m '

PS4='\033[0;32m$(date "+%Y%m%d %H:%M:%S.%N") $BASH_SOURCE:$LINENO]\033[0m '
set -euxo pipefail
