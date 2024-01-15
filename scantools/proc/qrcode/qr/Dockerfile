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

RUN python3 -m pip install --upgrade pip

#
# Builder stage.
#
FROM common as builder

RUN apt-get install -y --no-install-recommends --no-install-suggests \
    python3-dev \
    python3-setuptools \
    cmake \
    git   \
    build-essential \
    libgomp1 libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libzbar0


# Install dependencies.
COPY scripts/install_raybender.sh /tmp/
RUN bash /tmp/install_raybender.sh && rm /tmp/install_raybender.sh
# COPY scripts/install_pcdmeshing.sh /tmp/
# RUN apt-get install -y --no-install-recommends --no-install-suggests libgmp3-dev libmpfrc++-dev \
    # && bash /tmp/install_pcdmeshing.sh && rm /tmp/install_pcdmeshing.sh

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN git clone -b qrcode_squash https://github.com/pablospe/lamar-benchmark lamar \
    && cd lamar \
    && pip install -e .[scantools] --no-deps

#
# Runtime stage.
#
FROM common as runtime

RUN apt-get install -y --no-install-recommends --no-install-suggests \
    libgomp1 libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libzbar0

# Copy only the necessary files from the builder stage.

# COPY requirements.txt /tmp/
# RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY --from=builder /raybender /raybender

# # COPY --from=builder /usr/local/lib/python3.10/ /usr/local/lib/python3.10/
# # COPY --from=builder /usr/local/lib/python3.10/dist-packages/pcdmeshing* /usr/local/lib/python3.10/dist-packages/
# COPY --from=builder /usr/local/lib/python3.10/dist-packages/raybender* /usr/local/lib/python3.10/dist-packages/
# COPY --from=builder /raybender/dist-wheel /tmp/dist-wheel

# # Install lamar/scantools.
# # RUN pip install git+https://github.com/microsoft/lamar-benchmark.git --no-deps
# # RUN pip install git+https://github.com/pablospe/lamar-benchmark.git@qrcode_squash --no-deps

# RUN cd /tmp && whl_path=$(cat dist-wheel/whl_path.txt) && pip install $whl_path
# RUN rm -rfv /tmp/*


# # RUN python3 -m pip install --upgrade pip
# # RUN apt-get install -y --no-install-recommends --no-install-suggests \
# #     libgomp1 libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libzbar0
# # RUN git clone -b qrcode_squash https://github.com/pablospe/lamar-benchmark lamar \
# #     && cd lamar \
# #     && pip install -e .[scantools] --no-deps
