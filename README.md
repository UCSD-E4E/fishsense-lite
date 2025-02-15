# FishSense Lite

## Dependencies
* https://github.com/UCSD-E4E/pyFishSense
* https://github.com/UCSD-E4E/pyFishSenseDev

## Setup

### Install the dependencies

#### Supported Linux Distros

**Ubuntu 24.04:**

```console
sudo apt-get install -y build-essential git libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl
```

#### Unsupported Linux Distros
These Linux distros should function, but these steps are not regularly tested. YMMV.

**Fedora 41+:**

```console
sudo dnf install -y @development-tools git openssl-devel zlib-devel bzip2-devel sqlite-devel wget curl llvm ncurses-devel xz tk-devel libffi-devel xz-devel
```

**For RHEL 9 / Fedora 40**

```console
sudo dnf group install "Development Tools"
```

```console
sudo dnf install -y git openssl-devel zlib-devel bzip2-devel sqlite-devel wget curl llvm ncurses-devel xz tk-devel libffi-devel xz-devel
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

Source your `~/.bashrc`, install Python `3.12`, and poetry
```
source ~/.bashrc
pyenv install 3.12
```

Install rustup
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

In the root of this project, run the following.  If you do not have x86 with CUDA, you may need to delete poetry.lock.
```
pip install poetry
pip install --upgrade pip

poetry install
```

## Development Dependencies
We provide a `docker` container which has the dependencies pre-installed.  In order to use this, please ensure you have `docker` installed on your system.  When running Visual Studio Code, ensure that you use the option to reopen in the container.  This step will time for the intial setup.

## To Run Demo
Open the notebook in demo/pipeline.ipynb

## To run the CLI
To run the CLI to process data.  Note that this will not rerun files previously stored in `./output/results.db`.
```
poetry run python -m fishsense_lite process ~/data/**/*.ORF --lens-calibration ~/data/fsl-01d-lens-raw.pkg --laser-calibration ~/data/laser-calibration.pkg --output ./output/results.db
```
