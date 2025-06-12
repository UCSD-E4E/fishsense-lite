ARG IMAGE="ghcr.io/ucsd-e4e/fishsense:cpu"
FROM ${IMAGE}

ARG VERSION=v1.0.0
RUN pip install https://github.com/UCSD-E4E/fishsense-lite/archive/refs/tags/${VERSION}.tar.gz && \
    pip cache purge

CMD ["/bin/bash"]
