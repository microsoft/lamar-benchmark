FROM mcr.microsoft.com/mirror/docker/library/ubuntu:22.04 AS common

# Minimal toolings.
RUN apt-get update && \
    apt-get install -y \
        git \
        sudo \
        wget \
        bash

RUN apt-get install -y --no-install-recommends --no-install-suggests \
    python-is-python3 \
    python3-minimal \
    python3-pip


#
# Builder stage.
#
FROM common as builder

RUN apt-get install -y --no-install-recommends --no-install-suggests \
    python3-dev \
    python3-setuptools \
    cmake \
    git   \
    build-essential

# Install dependencies.
COPY scripts/install_raybender.sh /tmp/
COPY scripts/install_pcdmeshing.sh /tmp/
RUN bash /tmp/install_raybender.sh && rm /tmp/install_raybender.sh
RUN apt-get install -y --no-install-recommends --no-install-suggests libgmp3-dev libmpfrc++-dev \
    && bash /tmp/install_pcdmeshing.sh && rm /tmp/install_pcdmeshing.sh


#
# Runtime stage.
#
FROM common as runtime
# Copy only the necessary files from the builder stage.

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# COPY --from=builder /usr/local/lib/python3.10/ /usr/local/lib/python3.10/
COPY --from=builder /usr/local/lib/python3.10/dist-packages/pcdmeshing* /usr/local/lib/python3.10/dist-packages/
COPY --from=builder /usr/local/lib/python3.10/dist-packages/raybender* /usr/local/lib/python3.10/dist-packages/

# Install lamar/scantools.
RUN pip install git+https://github.com/microsoft/lamar-benchmark.git --no-deps
