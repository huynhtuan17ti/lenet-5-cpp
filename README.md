# Lenet-5 cpp
Lenet with cpp implementation

## Tech stack
- C/C++
- Bazel
- Eigen

## Getting started

```sh
# inference
bazel run //:inference # cpu
bazel run //:inference --config=cuda --//:conv_ver=v2 # cuda, using conv v2
bazel run //:inference --config=cuda --//:conv_ver=v3 # cuda, using conv v3 (default)

# test
bazel run //test:test --config=cuda --//:conv_ver=<ver>
```
