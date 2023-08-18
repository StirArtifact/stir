This file describes how to set up the environment to run the code.
Note that the following instructions may not work on any machine, as they have only been tested on limited number of machines (see [tested environments](REQUIREMENTS.md#tested-environment)).
If you encounter any issues, please contact us.

# Install with Docker (recommended)
The easiest way to install and run the code is to use Docker. 
To install Docker, follow the instructions here: https://docs.docker.com/get-docker/. 

## Step 1: Get the image
### Method 1: Import the image from a tar archive
Download the compressed tar archive of the Docker image from [the release page](https://github.com/StirArtifact/stir/releases/tag/fse2023) and import it with the following command:
```shell
docker load -i <path_to_tar_archive>
```
Using this method, the name and tag of the image will be `stir:latest`.
### Method 2: Build the image from the Dockerfile

You can run the following command in the root directory of this repo to build the Docker image.

```shell
docker build -t <image_name> .
```

## Step 2: Run the code
Start a shell in the container with the following commands:
```shell
docker run -it \
    --volume <absolute_path_to_save_trained_models>:/stir/models \
    <image_name> /bin/bash
```
If you need to run the code with a GPU, please add the `--gpus all` flag to the above command.

If Docker Compose is installed on your machine, the following shorter command can be used instead:
```shell
docker-compose run --rm --it stir
```
and if you need to run the code with a GPU:
```shell
docker-compose run --rm --it stir-gpu
```

After running the above commands, you should be in a shell in the container, where you can run the code as described in
[README.md](README.md). You may test that the environment is working by running the following command:
```shell
python main.py --help
```

# Install without Docker
If you do not want to use Docker, you can install the libraries required by the code on your machine, and
run the code directly.
To do so, you need to install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Then, you can install the libraries required by the code with the following commands:
```shell
conda env create -f environment.yml
conda activate stir
```
The code will automatically detect whether a GPU is available and use it if possible.

Then, you can run the code as described in [README.md](README.md). You may test that the environment is working by running the following command:
```shell
python main.py --help
```