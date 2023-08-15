# Hardware environment
The evaluation function of the code in this repository requires at least a computer with an x64 architecture CPU, 
while the training section can be run on the CPU or a GPU with support for NVIDIA CUDA. 

# Software environment
The code in this repository has been tested on both Windows and Linux platforms 
and can run on any environment equipped with the necessary Python libraries. 
The 'environment.yml' and 'requirements.txt' file in the root directory of the repository contains a list 
describing all the required Python library versions, 
and the 'Dockerfile' in the root directory can be used to build a Docker image with the required environment. 
For detailed information on setting up the environment, please refer to the 'INSTALL.md' file.

# Tested environment
All code in this repository was developed and tested on a Windows 10 desktop and an Ubuntu 18.04 Docker image on it, 
equipped with an 8-core Intel i7-7500 CPU of 3.40 GHz with 32GB memory, 
accelerated by a 12GB NVIDIA GeForce RTX 2080Ti GPU. 
The docker image was built using the 'Dockerfile' in the root directory of the repository. The Docker version is 24.0.2.
The Python environment was built using the 'environment.yml' files with Miniconda3 version 23.3.1.