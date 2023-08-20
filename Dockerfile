FROM ubuntu:18.04
SHELL ["/bin/bash", "-c"]

ARG PYTHON_VERSION=3.7.9
# Set the following arg if you need
ARG CONDA_INSTALL_URL=https://repo.anaconda.com/miniconda

WORKDIR /stir
COPY environment.yml /stir/environment.yml

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL -v -o ~/miniconda.sh -O ${CONDA_INSTALL_URL}/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda env create -f environment.yml && \
    /opt/conda/bin/conda clean -ya && \
    source /opt/conda/bin/activate stir && \
    echo "source /opt/conda/bin/activate stir" >> ~/.bash_profile && \
    echo "source /opt/conda/bin/activate stir" >> ~/.bashrc

COPY . /stir