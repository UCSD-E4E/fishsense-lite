ARG IMAGE="ghcr.io/ucsd-e4e/fishsense:cpu"
FROM ${IMAGE}

SHELL ["/bin/bash", "-c"]

RUN mkdir -p ~/.ssh && ssh-keyscan -H github.com >> ~/.ssh/known_hosts

ARG BUMP=1
RUN . ${HOME}/.cargo/env && pip install git+https://github.com/UCSD-E4E/fishsense-lite.git

ARG MAX_CPU=1
ARG MAX_GPU=1

RUN ${HOME}/.pyenv/shims/fsl generate-ray-config --max-cpu ${MAX_CPU} --max-gpu ${MAX_GPU}

USER root
ARG GID=1000
ARG UID=1000
ARG CUDA_GROUP_ID=1000
RUN groupmod -g ${GID} ubuntu && usermod -u ${UID} -g ${GID} ubuntu && groupadd -g ${CUDA_GROUP_ID} cuda
RUN sudo usermod -aG cuda ubuntu
USER ubuntu

CMD ["/bin/bash"]
