# Artifact for "Stir: Statical Type Inference for Incomplete Programs"

Stir is a novel two-stage approach for inferring types in incomplete programs that may be ill-formed, where whole-program syntactic analysis often fails. In the first stage, Stir predicts a type tag for each token by using neural networks, and consequently, infers all the simple types in the program. In the second stage, Stir refines the complex types for the tokens with predicted complex type tags. Unlike existing machine-learning-based approaches, which solve type inference as a classification problem, Stir reduces it to a sequence-to-graph parsing problem. This artifact contains the implementation and evaluation program of Stir, which can be used to reproduce the evaluation results, and can also serve as a standalone application for general use of the approach.

This artifact is organized as follows:
- `abstract.md`: this file.
- `README.md`: main document file.
- `INSTALL.md`: instructions for obtaining the artifact and setting up the environment.
- `REQUIREMENTS.md`: requirements for the hardware and software environment.
- `STATUS.md`: badges that this artifact applies for and the reasons for applying for them.
- `LICENSE`: license (MIT License) of the artifact.
- `main.py`: the main entry file.
- `first/`: the source code of the first stage of STIR.
- `second/`: the source code of the second stage of STIR.
- `data/`: the data used in the evaluation.
- `pretrained/`: the pretrained model used in the evaluation.
- `Dockerfile`: Dockerfile for building the Docker image with the software environment to reproduce the evaluation results.
- `environment.yml`: conda environment file for reproducing the evaluation results.

To begin with, please refer to [README.md](README.md) for a brief introduction to the artifact and instructions on how to use it.