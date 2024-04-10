# GGML Template

[![Build Actions Status](https://github.com/grazder/ggml_template/actions/workflows/build.yml/badge.svg)](https://github.com/grazder/ggml_template/actions/workflows/build.yml)

Template to start writing code using ggml

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
mkdir build; cd build
cmake ..; make
./model
```

## TODO

- [x] Basic FF example
- [ ] Python-CPP tests
- [ ] Trying on real model
- [ ] Adept template for real case usage
- [ ] Writing comments
- [ ] Add argparse for `model.cpp`
- [ ] Support FP16
- [ ] Quantization (?)

