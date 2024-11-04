FROM ghcr.io/ucsd-e4e/fishsense-lite:runtime-cuda

SHELL ["/bin/bash", "-c"]

ARG MAX_CPU=1
ARG MAX_GPU=1

RUN ${HOME}/.local/bin/fsl generate-ray-config --max-cpu ${MAX_CPU} --max-gpu ${MAX_GPU}

USER root
ARG GID=1000
ARG UID=1000
RUN groupmod -g ${GID} ubuntu && usermod -u ${UID} -g ${GID} ubuntu
USER ubuntu

CMD ["/bin/bash"]
