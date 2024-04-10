# GGML Template

Template to start writing code for your ggml model

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

