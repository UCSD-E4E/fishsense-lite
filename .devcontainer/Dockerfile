FROM nvidia/cuda:12.6.1-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"] 

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y sudo \
                        build-essential \
                        git \
                        libssl-dev \
                        zlib1g-dev \
                        libbz2-dev \
                        libreadline-dev \
                        libsqlite3-dev \
                        wget \
                        curl \
                        llvm \
                        libncurses5-dev \
                        libncursesw5-dev \
                        xz-utils \
                        tk-dev \
                        libffi-dev \
                        liblzma-dev \
                        python3-openssl \
                        libopencv-dev \
                        clang \
                        libclang-dev \
                        llvm \
                        python3-dev \
                        libpython3-dev \
                        ocl-icd-libopencl1 \
                        cmake \
                        pkg-config \
                        make \
                        ninja-build \
                        ocl-icd-libopencl1 \
                        ocl-icd-dev \
                        ocl-icd-opencl-dev \
                        libhwloc-dev \
                        clinfo \
                        dialog \
                        apt-utils \
                        libxml2-dev \
                        libclang-cpp-dev \
                        llvm-dev

# NVIDIA does not provide OpenCL passthru.
# POCL supports a CUDA-based OpenCL driver
RUN git clone https://github.com/pocl/pocl.git /pocl
WORKDIR /pocl
RUN git checkout v6.0
RUN mkdir build
WORKDIR /pocl/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/ -DENABLE_CUDA=ON .. && \
    make -j && \
    make install

RUN echo 'ubuntu ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

USER ubuntu
ENV HOME="/home/ubuntu"
RUN mkdir -p ${HOME}
WORKDIR ${HOME}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN pyenv install 3.11
RUN pyenv global 3.11
RUN pip install --upgrade pip
RUN pip install poetry

# We want access to all of the GPUs but do not want to expose any CPU devices
RUN echo "unset POCL_DEVICES" >> ${HOME}/.bashrc && \
  echo "gpu_count=\`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l\`" >> ${HOME}/.bashrc && \
  echo "cuda=cuda" >> ${HOME}/.bashrc && \
  echo "for i in \$(seq \$gpu_count); do" >> ${HOME}/.bashrc && \
  echo "  export POCL_DEVICES=\$cuda\$POCL_DEVICES" >> ${HOME}/.bashrc && \
  echo "  cuda=',cuda'" >> ${HOME}/.bashrc && \
  echo "done" >> ${HOME}/.bashrc
