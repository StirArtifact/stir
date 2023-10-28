FROM ubuntu:23.10
SHELL ["/bin/bash", "-c"]

ARG PYTHON_VERSION=3.7.9
# Set the following arg if you need
# ARG UBUNTU_SOURCE_HOST=archive.ubuntu.com
ARG UBUNTU_SOURCE_HOST=mirrors.ustc.edu.cn
# ARG PYPI_REGISTRY_URL=https://pypi.org/simple
ARG PYPI_REGISTRY_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# ARG CONDA_INSTALL_URL=https://repo.anaconda.com/miniconda
ARG CONDA_INSTALL_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda
# ARG CONDA_CHANNEL_URL=https://repo.anaconda.com
ARG CONDA_CHANNEL_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda
# ARG PYTORCH_CHANNEL_URL=https://conda.anaconda.org
ARG CUSTOM_CHANNEL_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt,sharing=locked \
    sed -i "s@//.*archive.ubuntu.com@//${UBUNTU_SOURCE_HOST}@g" /etc/apt/sources.list && \ 
    sed -i "s/security.ubuntu.com/${UBUNTU_SOURCE_HOST}/g" /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl && \
    sed -i 's/http:/https:/g' /etc/apt/sources.list && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL -v -o ~/miniconda.sh -O ${CONDA_INSTALL_URL}/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    echo "channels:" > /root/.condarc && \
    echo "  - defaults" >> /root/.condarc && \
    echo "show_channel_urls: true" >> /root/.condarc && \
    echo "default_channels:" >> /root/.condarc && \
    echo "  - ${CONDA_CHANNEL_URL}/pkgs/main" >> /root/.condarc && \
    echo "custom_channels:" >> /root/.condarc && \
    echo "  pytorch: ${CUSTOM_CHANNEL_URL}" >> /root/.condarc && \
    echo "  conda-forge: ${CUSTOM_CHANNEL_URL}" >> /root/.condarc
    # echo "  pytorch: ${CUSTOM_CHANNEL_URL}" >> /root/.condarc

WORKDIR /stir
COPY environment.yml /stir/environment.yml

RUN /opt/conda/bin/conda env create -f environment.yml && \
    /opt/conda/bin/conda clean -ya && \
    source /opt/conda/bin/activate stir && \
    echo "source /opt/conda/bin/activate stir" >> ~/.bash_profile && \
    echo "source /opt/conda/bin/activate stir" >> ~/.bashrc

COPY . /stir