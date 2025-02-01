FROM ghcr.io/ucsd-e4e/fishsense-lite:cuda

SHELL ["/bin/bash", "-c"]

RUN mkdir -p ~/.ssh && ssh-keyscan -H github.com >> ~/.ssh/known_hosts

RUN pip install pipx
RUN pipx ensurepath

ARG BUMP=0
RUN . ${HOME}/.cargo/env && pipx install git+https://github.com/UCSD-E4E/fishsense-lite.git

ARG MAX_CPU=1
ARG MAX_GPU=1

RUN ${HOME}/.local/bin/fsl generate-ray-config --max-cpu ${MAX_CPU} --max-gpu ${MAX_GPU}

USER root
ARG GID=1000
ARG UID=1000
RUN groupmod -g ${GID} ubuntu && usermod -u ${UID} -g ${GID} ubuntu
USER ubuntu

CMD ["/bin/bash"]
