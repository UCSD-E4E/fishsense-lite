ARG IMAGE=ghcr.io/ucsd-e4e/fishsense-lite:base-cpu
FROM $IMAGE

RUN echo 'ubuntu ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

USER ubuntu
ENV HOME="/home/ubuntu"
RUN mkdir -p ${HOME}
WORKDIR ${HOME}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN echo ". \"\$HOME/.cargo/env\"" >> ${HOME}/.bashrc

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN echo "export PYENV_ROOT=\"\$HOME/.pyenv\"" >> ${HOME}/.bashrc && \
	echo "[[ -d \$PYENV_ROOT/bin ]] && export PATH="\$PYENV_ROOT/bin:\$PATH"" >> ${HOME}/.bashrc && \
	echo "eval \"\$(pyenv init -)\"" >> ${HOME}/.bashrc && \
	echo "\"$(pyenv virtualenv-init -)\"" >> ${HOME}/.bashrc

RUN pyenv install 3.11
RUN pyenv global 3.11
RUN pip install --upgrade pip && pip cache purge

CMD ["/bin/bash"]