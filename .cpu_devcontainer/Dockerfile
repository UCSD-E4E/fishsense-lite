FROM ubuntu:24.04

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
                        llvm

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
