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

# ceres dependency
RUN apt-get install -y \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev

### CMake 3.16, Eigen 3.3, glog 0.3.5, SuiteSparse 4.5.6 recommended
RUN git clone https://github.com/ceres-solver/ceres-solver.git && \ 
    cd ceres-solver && \
    git checkout tags/2.2.0 && \
    mkdir build_ && \
    cd build_ &&\
    cmake .. && \
    make -j3 && \
    make install 

# sudo should be installed upper stage -> when usign docker ...
RUN git clone --recursive https://github.com/stevenlovegrove/Pangolin.git && \
    apt-get -y install sudo \
               libgl-dev \
               libglew-dev&&\
    cd Pangolin && \
    git checkout tags/v0.8 && \
    ./scripts/install_prerequisites.sh --dry-run recommended && \
    cmake -B build &&\
    cmake --build build 

