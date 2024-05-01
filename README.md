# 💽 template.cpp

[![Build Actions Status](https://github.com/grazder/ggml_template/actions/workflows/build.yml/badge.svg)](https://github.com/grazder/ggml_template/actions/workflows/build.yml)
[![Tests Actions Status](https://github.com/grazder/ggml_template/actions/workflows/tests.yml/badge.svg)](https://github.com/grazder/ggml_template/actions/workflows/tests.yml)

A template for getting started writing code using [`GGML`](https://github.com/ggerganov/ggml.git).


## Features

- Simple function with linear layer added
- Export model weights to `.gguf` format
- Compare your python and GGML code using tests

## Usage

To use this template, follow these steps:

1. Clone the repository: `git clone https://github.com/grazder/ggml_template.git --recursive`
2. Navigate to the project directory: `cd ggml_template`
3. Export model weights to `.gguf` format: `python weights_export/export_model_weights.py`
4. Build the project: 
    ```
    mkdir build
    cd build
    cmake ..
    make
    ```
5. Run the project: `./example/main`
6. Run tests: `python -m pytest tests/test.py`

## Start rewriting your model to GGML

1. Export your model to GGUF format. Example in `weights_export/export_model_weights.py`
2. Load your GGUF file into CPP code. Example in `template.cpp` - `load_weigths` and `load_hparams` functions
3. Write inference code for your model. Example in `template.cpp` - `forward` and `compute`.
4. Write usage example. Example in `example/main.cpp`.
5. Write python bindings for your model. Example in `tests/bindings.cpp`
6. Write tests for python and cpp code comparison. Example in `tests/test.py`.

## TODO

- [x] Basic FF example
- [x] Python-CPP tests
- [x] Add GGUF
- [x] Make cleaning
- [ ] Try on real model
- [ ] Adapt template for real case usage
- [ ] Write comments
- [ ] Add argparse for `model.cpp`
- [ ] Support FP16
- [ ] Quantization (?)

