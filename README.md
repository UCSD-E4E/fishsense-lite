# FishSense Lite

## To Run Demo
On Ubuntu 22.04, install
```
sudo apt-get install -y build-essential git libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl
```

Then, make sure you have `pyenv` installed.
```
curl https://pyenv.run | bash
```

Then, add the following to your `~/.bashrc`
```
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Source your `~/.bashrc`, install Python `3.11`, and poetry
```
source ~/.bashrc
pyenv install 3.11

pip install poetry
pip install --upgrade pip
```

In the root of this project, run the following.  If you do not have x86 with CUDA, you may need to delete poetry.lock.
```
poetry install --no-root
poetry run python -m pip install git+https://github.com/facebookresearch/detectron2.git@3c7bb714795edc7a96c9a1a6dd83663ecd293e36 –no-build-isolation
```