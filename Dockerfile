FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

# install basic module 
RUN apt-get install -y \
    build-essential \
    vim \
    cmake \
    git \
    python3 \
    python3-pip

# install additional library
RUN apt-get install -y \
    libgtest-dev \
    libgoogle-glog-dev \
    libopencv-dev \
    libeigen3-dev
