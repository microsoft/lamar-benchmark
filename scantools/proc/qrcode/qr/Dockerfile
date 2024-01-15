FROM mcr.microsoft.com/mirror/docker/library/ubuntu:22.04 AS common

# Minimal toolings.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        sudo \
        wget \
        bash

RUN apt-get install -y --no-install-recommends --no-install-suggests \
    python-is-python3 \
    python3-minimal \
    python3-pip

RUN apt-get install -y --no-install-recommends --no-install-suggests \
    libgomp1 libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libzbar0

RUN python3 -m pip install --upgrade pip

RUN git clone https://github.com/microsoft/lamar-benchmark lamar
# WORKDIR /lamar


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

# Install raybender.
COPY scripts/build_raybender.sh /tmp/
RUN bash /tmp/build_raybender.sh && rm /tmp/build_raybender.sh

# Install pcdmeshing.
COPY scripts/build_pcdmeshing.sh /tmp/
RUN bash /tmp/build_pcdmeshing.sh && rm /tmp/build_pcdmeshing.sh

# Install hloc.
COPY scripts/build_pcdmeshing.sh /tmp/
RUN bash /tmp/build_pcdmeshing.sh && rm /tmp/build_pcdmeshing.sh


#
# Scantools stage.
#
FROM common as scantools

# Install raybender.
COPY --from=builder /raybender/embree-3.12.2/lib /raybender/embree-3.12.2/lib
COPY --from=builder /raybender/dist-wheel /tmp/dist-wheel
RUN cd /tmp && whl_path=$(cat dist-wheel/whl_path.txt) && pip install $whl_path
RUN rm -rfv /tmp/*

# Install pcdmeshing.
COPY --from=builder /pcdmeshing/dist-wheel /tmp/dist-wheel
RUN sudo apt-get install -y --no-install-recommends --no-install-suggests \
        libmpfrc++-dev
RUN cd /tmp && whl_path=$(cat dist-wheel/whl_path.txt) && pip install $whl_path
RUN rm -rfv /tmp/*

RUN pip install astral==3.2 beautifulsoup4==4.12.2 lxml==4.9.2 \
    matplotlib open3d==0.18.0 opencv-python==4.7.0.72 plyfile==1.0.3 \
    pytijo==0.0.2 pyzbar-upright==0.1.8 scipy==1.11.4 --no-deps
RUN cd lamar && pip install -e .[scantools] --no-deps



#
# Lamar stage.
#
FROM scantools as lamar

# Install hloc.
COPY --from=builder /hloc/dist-wheel /tmp/dist-wheel
RUN cd /tmp && whl_path=$(cat dist-wheel/whl_path.txt) && pip install $whl_path
RUN rm -rfv /tmp/*

RUN cd /lamar && pip install -e .
