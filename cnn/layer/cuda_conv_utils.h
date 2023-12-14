#pragma once
#include <iostream>
#include <cstdint>

class CudaConv {
 public:
  CudaConv() {} 

  void SetInMatrix(size_t channel_in, size_t width_in, size_t height_in);

  void SetKernel(size_t kernel_size);

  void SetOutMatrix(size_t channel_out, size_t width_out, size_t height_out);

  void Launch(const float* in_matrix, const float* kernel, float* out_matrix);

 private:
  size_t channel_in_, width_in_, height_in_; 
  size_t kernel_size_;
  size_t channel_out_, width_out_, height_out_;
};
