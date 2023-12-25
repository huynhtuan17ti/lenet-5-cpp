#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

const size_t N_STREAMS = 16;

class CudaConv {
 public:
  CudaConv();

  ~CudaConv();

  void InitKernelParams(float* kernel);

  void InitBiasParams(float* bias);

  void SetInMatrix(size_t channel_in, size_t width_in, size_t height_in);

  void SetKernel(size_t kernel_size);

  void SetOutMatrix(size_t channel_out, size_t width_out, size_t height_out);

  void LaunchOnOneSample(const float* in_matrix, float* out_matrix);

  void LaunchOnSamples(const float* in_matrix, float* out_matrix, size_t n_sample);

 private:
  size_t channel_in_, width_in_, height_in_;
  size_t kernel_size_;
  size_t channel_out_, width_out_, height_out_;

  float* d_kernel_;
  float* d_bias_;
  cudaStream_t streams_[N_STREAMS];
};
