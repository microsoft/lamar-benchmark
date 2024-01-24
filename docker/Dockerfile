# FROM mcr.microsoft.com/mirror/docker/library/ubuntu:22.04 AS common
ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.3.1
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as common

# Minimal toolings.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        bash \
        git \
        python-is-python3 \
        python3-minimal \
        python3-pip \
        sudo \
        wget

RUN python3 -m pip install --upgrade pip

RUN git clone --depth 1 --recursive https://github.com/microsoft/lamar-benchmark lamar

#
# Builder stage.
#
FROM common as builder

RUN apt-get install -y --no-install-recommends --no-install-suggests \
    build-essential \
    cmake \
    libeigen3-dev \
    python3-dev \
    python3-setuptools

# Build raybender.
COPY scripts/build_raybender.sh /tmp/
RUN bash /tmp/build_raybender.sh && rm /tmp/build_raybender.sh

# Build pcdmeshing.
COPY scripts/build_pcdmeshing.sh /tmp/
RUN bash /tmp/build_pcdmeshing.sh && rm /tmp/build_pcdmeshing.sh

# Build hloc.
COPY scripts/build_hloc.sh /tmp/
RUN bash /tmp/build_hloc.sh && rm /tmp/build_hloc.sh

#
# Scantools stage.
#
FROM common as scantools

RUN apt-get install -y --no-install-recommends --no-install-suggests \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libzbar0

# Install raybender.
COPY --from=builder /raybender/embree-3.12.2/lib /raybender/embree-3.12.2/lib
COPY --from=builder /raybender/dist-wheel /tmp/dist-wheel
RUN cd /tmp && whl_path=$(cat dist-wheel/whl_path.txt) && python3 -m pip install $whl_path
RUN rm -rfv /tmp/*

# Install pcdmeshing.
COPY --from=builder /pcdmeshing/dist-wheel /tmp/dist-wheel
RUN sudo apt-get install -y --no-install-recommends --no-install-suggests \
        libmpfrc++-dev
RUN cd /tmp && whl_path=$(cat dist-wheel/whl_path.txt) && python3 -m pip install $whl_path
RUN rm -rfv /tmp/*

RUN python3 -m pip install --no-deps \
        astral==3.2 \
        beautifulsoup4==4.12.2 \
        lxml==4.9.2 \
        matplotlib \
        open3d==0.18.0 \
        opencv-python==4.7.0.72 \
        plyfile==1.0.3 \
        pytijo==0.0.2 \
        pyzbar-upright==0.1.8 \
        scipy==1.11.4
RUN cd lamar && python3 -m pip install -e .[scantools] --no-deps

#
# pyceres-builder stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as pyceres-builder

ARG COLMAP_VERSION=3.9.1
ARG CUDA_ARCHITECTURES=70
ENV CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        python-is-python3 \
        python3-minimal \
        python3-pip \
        python3-dev \
        python3-setuptools

# Build Ceres solver.
RUN apt-get install -y --no-install-recommends --no-install-suggests wget && \
    wget "http://ceres-solver.org/ceres-solver-2.1.0.tar.gz" && \
    tar zxf ceres-solver-2.1.0.tar.gz && \
    mkdir ceres-build && \
    cd ceres-build && \
    cmake ../ceres-solver-2.1.0 -GNinja \
        -DCMAKE_INSTALL_PREFIX=/ceres_installed && \
    ninja install
RUN cp -r /ceres_installed/* /usr/local/

# Install Colmap.
RUN wget "https://github.com/colmap/colmap/archive/refs/tags/${COLMAP_VERSION}.tar.gz" -O colmap-${COLMAP_VERSION}.tar.gz && \
    tar zxvf colmap-${COLMAP_VERSION}.tar.gz && \
    mkdir colmap-build && \
    cd colmap-build && \
    cmake ../colmap-${COLMAP_VERSION} -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap_installed && \
    ninja install
RUN cp -r /colmap_installed/* /usr/local/

# Build pyceres.
# Todo: Use the official pyceres wheel when it is available.
RUN git clone --depth 1 -b update-colmap-3.9-and-docker-image --recursive https://github.com/pablospe/pyceres
RUN cd pyceres && \
    pip wheel --no-deps -w dist-wheel . && \
    whl_path=$(find dist-wheel/ -name "*.whl") && \
    echo $whl_path >dist-wheel/whl_path.txt

#
# pyceres stage.
#
FROM scantools as pyceres

COPY --from=pyceres-builder /ceres_installed/ /usr/local/
COPY --from=pyceres-builder /colmap_installed/ /usr/local/

# Install pyceres.
COPY --from=pyceres-builder /pyceres/dist-wheel /tmp/dist-wheel
RUN cd /tmp && whl_path=$(cat dist-wheel/whl_path.txt) && python3 -m pip install $whl_path
RUN rm -rfv /tmp/*

#
# lamar stage.
#
FROM pyceres as lamar

# Install hloc.
COPY --from=builder /hloc/dist-wheel /tmp/dist-wheel
RUN cd /tmp && whl_path=$(cat dist-wheel/whl_path.txt) && python3 -m pip install $whl_path
RUN rm -rfv /tmp/*

# Note: The dependencies listed in pyproject.toml also include pycolmap and
# pyceres. These are already installed in previous Docker stages. Attempting to
# compile them in this stage will lead to failure due to missing necessary
# development dependencies.
RUN python3 -m pip install --no-deps \
        h5py==3.10.0 \
        numpy==1.26.3 \
        torch>=1.1 \
        tqdm>=4.36.0

RUN cd /lamar && python3 -m pip install -e .  --no-deps
