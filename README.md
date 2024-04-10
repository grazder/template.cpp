# 💽 template.cpp

[![Build Actions Status](https://github.com/grazder/ggml_template/actions/workflows/build.yml/badge.svg)](https://github.com/grazder/ggml_template/actions/workflows/build.yml)

A template for getting started writing code using [`GGML`](https://github.com/ggerganov/ggml.git).

## Install

```
git clone https://github.com/ggerganov/ggml.git
```

## Creating `.bin` weights

```
python save_model_weights.py
```

## Build

```
mkdir build
cd build
cmake ..
make
./model
```

## TODO

- [x] Basic FF example
- [ ] Python-CPP tests
- [ ] Trying on real model
- [ ] Adapt template for real case usage
- [ ] Writing comments
- [ ] Add argparse for `model.cpp`
- [ ] Support FP16
- [ ] Quantization (?)

