FROM ubuntu:18.04
SHELL ["/bin/bash", "-c"]

ARG PYTHON_VERSION=3.7.9
# Set the following arg if you need
# ARG PYPI_REGISTRY_URL=https://pypi.org/simple
ARG PYPI_REGISTRY_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# ARG CONDA_INSTALL_URL=https://repo.anaconda.com/miniconda
ARG CONDA_INSTALL_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda

WORKDIR /stir
COPY environment.yml /stir/environment.yml

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt,sharing=locked \
    # remember to delete the following lines if you are not in China
    sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list && \ 
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    # sed -i 's/http:/https:/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL -v -o ~/miniconda.sh -O ${CONDA_INSTALL_URL}/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    echo $'channels:\n\
  - defaults\n\
show_channel_urls: true\n\
default_channels:\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\n\
custom_channels:\n\
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/' > /root/.condarc && \
#    /opt/conda/bin/conda create -y --name stir python=${PYTHON_VERSION} && \
#    /opt/conda/bin/conda install -y --name stir ipython && \
#    /opt/conda/bin/conda install -y --name stir pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch && \
    /opt/conda/bin/conda env create -f environment.yml && \
    /opt/conda/bin/conda clean -ya && \
    source /opt/conda/bin/activate stir && \
    echo "source /opt/conda/bin/activate stir" >> ~/.bash_profile && \
    echo "source /opt/conda/bin/activate stir" >> ~/.bashrc

COPY . /stir