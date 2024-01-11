# Lenet-5 cpp
Lenet 5 with cpp implementation and cuda optimazation in convolutional layers.

## Tech stack
- C/C++
- Bazel
- Eigen
- Cuda

## Getting started

```sh
# evaluate
bazel run //:evaluate # cpu
bazel run //:evaluate --config=cuda --//:conv_ver=v1 # cuda, using conv v1
bazel run //:evaluate --config=cuda --//:conv_ver=v2 # cuda, using conv v2
bazel run //:evaluate --config=cuda --//:conv_ver=v3 # cuda, using conv v3 (default)

# test
bazel run //test:test --config=cuda --//:conv_ver=<ver>
```

## Cuda optimization
There are 3 versions of cuda optimization:
~~~~~ so lazy to write :sleeping: ~~~~~~

## Reference
- [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp)
