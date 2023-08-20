# Stir: Statical Type Inference for Incomplete Programs
This repo contains the implementation and evaluation program of the ESEC/FSE 2023 paper 
entitled "Statistical Type Inference for Incomplete Program". It can be used to reproduce the evaluation results of the paper, and can also serve as a standalone tool for general usage of the algorithms discussed in the paper.

The outline of this document is as follows.
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Obtaining the Artifact](#obtaining-the-artifact)
  - [Setting up the Environment](#setting-up-the-environment)
- [Reproduce the Evaluation Results](#reproduce-the-evaluation-results)
  - [TL;DR: The fastest way to check the evaluation results](#tldr-the-fastest-way-to-check-the-evaluation-results)
  - [Detailed Usage of the `python main.py eval` Subcommand](#detailed-usage-of-the-python-mainpy-eval-subcommand)
- [Use as a standalone application](#use-as-a-standalone-application)
  - [Training](#training)
- [Acquire the data](#acquire-the-data)

## Getting Started

### Requirements
For the hardware and software requirements of the artifact, please refer to [REQUIREMENTS.md](REQUIREMENTS.md).

### Obtaining the Artifact

The artifact is available at [GitHub](https://github.com/StirArtifact/stir/tree/fse2023). Users can obtain the artifact by cloning the repository or downloading the source code as a compressed archive on the webpage.

### Setting up the Environment
For detailed instructions on setting up the environment, please refer to [INSTALL.md](INSTALL.md).


## Reproduce the Evaluation Results
The `main.py` file is the main entry file of the artifact, which provides a command line interface to evaluate the 
artifact. The detailed usage of the `main.py` file is described as follows. A brief help message can also be obtained by
executing the following command in the root directory of the artifact:

```shell
python main.py --help

```
To reproduce the evaluation results, follow the instructions below.

### TL;DR: The fastest way to check the evaluation results

#### Research Question 1
```shell
python main.py eval RQ1
```
The output of the command corresponds to the results of the first research question in the paper, as depicted in Table 7.
It will prepare some intermediate files, infer type tags, and print the results for each model.
#### Research Question 2 and 3
```shell
python main.py eval RQ2,RQ3
```
The output of the command corresponds to the results of the second and third research questions in the paper, as depicted in Table 8 and 9. 
It will prepare some intermediate files, generate complex types, and print the evaluation results for each model.
For each model, there will be a long process to calculate the evaluation result. You might need to wait for a period of time (~90 min in our test environment with GPU) for the whole evaluation to complete.

### Detailed Usage of the `python main.py eval` Subcommand

The `eval` subcommand of the `main.py` file should be used in the following format.

```shell
python main.py eval <RQ> [--data DATA --model MODEL]
```

The `<RQ>` argument specifies the research question to be evaluated, and the optional `--data` and `--model`
arguments specify the data and the pretrained model to be used in the evaluation. The `<RQ>` argument can be one of `RQ1` or
`RQ2,RQ3`, which correspond to the research questions in the paper. Note that the evaluation of `RQ2` and `RQ3` are
very time-consuming. Therefore, we combine them into one command, as their processes are similar.

The `--data` argument specifies the location of data files to be
used in the evaluation, which should be organized as follows.
```text
data/
├── simple
│   ├── test
│   │   ├── first_stage_test_files
│   │   ├── ...
│   first_stage_train_files
│   ...
├── complex
│   ├── test
│   │   ├── second_stage_test_files
│   │   ├── ...
│   second_stage_train_files
│   ...
``` 
The `simple` and `complex` directories contain the data for the evaluation of RQ1 and the grouped evaluation of RQ2 and RQ3,
respectively. The `test` directories contain the data for the test set, and the other directories contain the data for
the training set. The default value of the `--data` argument is `data`, which means that the data is in the `data/` directory.

The `--model` argument specifies the pretrained model to be used in the evaluation, whose available options depend on
the `<RQ>` argument. To evaluate multiple pretrained models at one time, separate the model names with commas without
spaces (e.g., `--model STIR,STIR_A`).
The available models for each research question are as follows:
- `RQ1`: `STIR`, `STIR_A`, `DeepTyper`, `TRAINED`
- `RQ2,RQ3`: `STIR`, `STIR_OT`, `STIR_DT`, `STIR_GT`, `TRAINED`, `TRAINED_OT`, `TRAINED_DT`, `TRAINED_GT`

where `STIR`, `STIR_A`, `DeepTyper`, `STIR_OT`, `STIR_DT` and `STIR_GT` correspond to the models described in each
research question, and the `TRAINED` series corresponds to the model trained by users (this will be explained later). `STIR_OT` and `TRAINED_OT` will not 
be evaluated for RQ3, as explained in the paper. The default value of the `--model` argument is `STIR,STIR_A,DeepTyper` for `RQ1` and `STIR,STIR_OT,STIR_DT,STIR_GT` for `RQ2,RQ3`.

For example, to reproduce the results of the first research question, the following command can be used:
```shell
python main.py eval RQ1 --data data --model STIR,STIR_A,DeepTyper
```
or with the default values of the `--data` and `--model` arguments:
```shell
python main.py eval RQ1
```

Note that the evaluation process may take a long time to complete, especially for the `RQ2` and `RQ3` research
questions. This is because the generation of some intermediate files is time-consuming. In order to reduce the
evaluation time, a cache mechanism is used in the evaluation process. The intermediate files as well as the hash values
of the combination of them and the data source used in their generation are cached in the `out/` directory in the
`out/` directory in the respective model directory. If the hash value corresponding to the file to be generated is not changed, the
file will not be generated again. Otherwise, the file will be generated again. 

## Use Your Own Data

### Training
To train a model by yourself, run the following command in the root directory of the artifact:
```shell
python main.py train <STAGE> [--data DATA]
```
where the `<STAGE>` argument specifies the stage to be trained, and the optional `--data` argument
specify the data to be used in the training. The `<STAGE>` argument can be one of `first`
and `second`, which correspond to the first or the second stage of the approach described in the paper. Note that to train the second stage, the first stage must be trained first.

The `--data` argument specifies the data to be used in the training, which should be organized as follows.
```text
user_data/
├── simple
│   ├── test
│   │   ├── first_stage_test_files
│   │   ├── ...
│   first_stage_train_files
│   ...
├── complex
│   ├── test
│   │   ├── second_stage_test_files
│   │   ├── ...
│   second_stage_train_files
│   ...
```
where the `simple` and `complex` directories contain the data for the first and second stages of the training,
respectively. The `test` directories contain the data for the test set. The default value of the `--data` argument is
`user_data`.

For example, to train the first stage of the approach described in the paper, run the following command in the root
directory of the artifact:
```shell
python main.py train first --data data
```

### Testing
To test a model on your own, run the following command in the root directory of the artifact:
```shell
python main.py test <STAGE> [--data DATA --model MODEL]
```
where the `<STAGE>` argument specifies the stage to be tested, and the optional `--data` and `--model` arguments
specify the data and the pretrained model to be used in the testing. The `<STAGE>` argument can be one of `first`
and `second`, which correspond to the first or the second stage of the approach described in the paper. 
This command is complementary to the `train` command. Its primary purpose is to facilitate the testing of your self-trained model using your own test data, although it can also be used to test our pretrained models.
The difference between this command and the `eval` subcommand is that 
1. The `--data` parameter defaults to `user_data` in the `test` command.
2. the `test` command uses the self-trained models by default, and
3. The directory containing intermediate files differs between the two commands.

## Acquire the data
The data used in our evaluation and training process can be obtained from [the release page](https://github.com/yuanmt/stir/releases) as a compressed tar archive. To use the data, extract the archive to the `data/` directory in the root directory of the artifact.
If you want to use your own
data, you can place it in the `data/` directory. Each of the file in the `data/` directory should be a plain text file
containing tokens and corresponding types of a program, where each line of the file contains a token and its type in the
corresponding code file, separated by a tab character. The rules for the token and type are as follows.
- The type tag for variables, constants and functions used in first stage should be their type names.
- The type label for variables, constants and functions used in second stage should be their type expressions. The
  expression of simple types should be the same as their type names, and the expression of complex types should be
  enclosed in parentheses and separated by commas, e.g., `struct(int,int)`, `(int,int)->(int)`, `*(int)`. 
- The type tag for any other tokens should be a special type tag which is not a valid type name or type expression
  to distinguish them from variables, constants and functions, e.g., `O`.
- As mentioned in the paper, recursive types are not supported in the current version of STIR. Therefore, recursive
  types have to be treated specifically. The type label for recursive types should just include the category of the 
  type and ends the list of its children with a `` ` ``, e.g., ``struct(`)``.

Files longer than 1000 tokens will be ignored in the training process. The data used in our evaluation and training process is obtained from
[GNU](https://www.gnu.org/), processed by a modified version of [Clang](https://clang.llvm.org/). Users may create
their own data by themselves, as long as the generated data conforms to the above rules. 