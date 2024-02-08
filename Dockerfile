ARG UBUNTU_VERSION=22.04
FROM mcr.microsoft.com/mirror/docker/library/ubuntu:${UBUNTU_VERSION} AS common

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

ADD . /lamar

#
# Builder stage.
#
FROM common AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        build-essential \
        cmake \
        libeigen3-dev \
        python3-dev \
        python3-setuptools

# Build raybender.
COPY docker/scripts/build_raybender.sh /tmp/
RUN bash /tmp/build_raybender.sh && rm /tmp/build_raybender.sh

# Build pcdmeshing.
COPY docker/scripts/build_pcdmeshing.sh /tmp/
RUN bash /tmp/build_pcdmeshing.sh && rm /tmp/build_pcdmeshing.sh

# Build hloc.
COPY docker/scripts/build_hloc.sh /tmp/
RUN bash /tmp/build_hloc.sh && rm /tmp/build_hloc.sh

#
# Scantools stage.
#
FROM common AS scantools

RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
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
FROM mcr.microsoft.com/mirror/docker/library/ubuntu:${UBUNTU_VERSION} AS pyceres-builder

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libeigen3-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libgtest-dev \
        libatlas-base-dev \
        libsuitesparse-dev \
        python-is-python3 \
        python3-minimal \
        python3-pip \
        python3-dev \
        python3-setuptools

# Install Ceres.
RUN apt-get install -y --no-install-recommends --no-install-suggests wget && \
    wget "http://ceres-solver.org/ceres-solver-2.1.0.tar.gz" && \
    tar zxf ceres-solver-2.1.0.tar.gz && \
    mkdir ceres-build && \
    cd ceres-build && \
    cmake ../ceres-solver-2.1.0 -GNinja \
        -DCMAKE_INSTALL_PREFIX=/ceres_installed && \
    ninja install
RUN cp -r /ceres_installed/* /usr/local/

# Build pyceres.
RUN git clone --depth 1 --recursive https://github.com/cvg/pyceres
RUN python3 -m pip install --upgrade pip
RUN cd pyceres && \
    pip wheel . --no-deps -w dist-wheel -vv && \
    whl_path=$(find dist-wheel/ -name "*.whl") && \
    echo $whl_path >dist-wheel/whl_path.txt

#
# pyceres stage.
#
FROM scantools as pyceres

# Install minimal runtime dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libgoogle-glog0v5 \
        libspqr2 \
        libcxsparse3 \
        libatlas3-base \
        python-is-python3 \
        python3-minimal \
        python3-pip

# Copy installed library in the builder stage.
COPY --from=pyceres-builder /ceres_installed/ /usr/local/

# Install pyceres.
COPY --from=pyceres-builder /pyceres/dist-wheel /tmp/dist-wheel
RUN pip install --upgrade pip
RUN cd /tmp && whl_path=$(cat dist-wheel/whl_path.txt) && pip install $whl_path
RUN rm -rfv /tmp/*

#
# lamar stage.
#
FROM pyceres as lamar

# Install hloc.
COPY --from=builder /hloc/dist-wheel /tmp/dist-wheel
RUN cd /tmp && whl_path=$(cat dist-wheel/whl_path.txt) && python3 -m pip install $whl_path
RUN rm -rfv /tmp/*

# Note: The dependencies listed in pyproject.toml also include pyceres, already
# installed in previous Docker stages. Attempting to compile it in this stage
# will lead to failure due to missing necessary development dependencies.
# Therefore, we replicate the dependencies here, excluding pyceres
RUN python3 -m pip install --no-deps \
        h5py==3.10.0 \
        numpy==1.26.3 \
        torch>=1.1 \
        tqdm>=4.36.0 \
        pycolmap==0.6.0

RUN cd /lamar && python3 -m pip install -e .  --no-deps
