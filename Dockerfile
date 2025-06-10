ARG IMAGE="ghcr.io/ucsd-e4e/fishsense:cpu"
FROM ${IMAGE}

RUN pip install git+https://github.com/UCSD-E4E/fishsense-lite.git@main && pip cache purge

CMD ["/bin/bash"]
