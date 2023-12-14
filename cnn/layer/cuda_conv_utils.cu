#include "cuda_conv_utils.h"
#include <stdio.h>
#include <stdlib.h>

const dim3 BLOCK_SIZE(8, 8);

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

__global__ void conv_kernel(float *in_matrix, size_t in_channel, size_t in_width, size_t in_height, 
                            float *kernel, int kernel_size, 
                            float *out_matrix, size_t out_channel, size_t out_width, size_t out_height) {
	size_t out_row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t out_col = blockIdx.x * blockDim.x + threadIdx.x;

	if (out_row < out_height && out_col < out_width) {
		size_t out_id = out_row * out_width + out_col;
    for(size_t in_row = out_row; in_row < out_row + kernel_size; ++in_row)
      for(size_t in_col = out_col; in_col < out_col + kernel_size; ++in_col) {
        size_t kernel_row = in_row - out_row, kernel_col = in_col - out_col;
        size_t kernel_id = kernel_row * kernel_size + kernel_col;

        size_t in_id = in_row * in_width + in_col; 
        for(size_t out_channel_id = 0; out_channel_id < out_channel; ++out_channel_id) {
          float res = 0;
          for(int in_channel_id = 0; in_channel_id < in_channel; ++in_channel_id) {
            float kernel_val = kernel[out_channel * (in_channel * kernel_id + in_channel_id) + out_channel_id];
            res += kernel_val * in_matrix[in_channel * in_id + in_channel_id];
          }
          out_matrix[out_channel * out_id + out_channel_id] = static_cast<uint8_t>(res);
        }
      }
	}
}

void CudaConv::SetInMatrix(size_t channel_in, size_t width_in, size_t height_in) {
  channel_in_ = channel_in;
  width_in_ = width_in;
  height_in_ = height_in;
}

void CudaConv::SetKernel(size_t kernel_size) {
  kernel_size_ = kernel_size;
}

void CudaConv::SetOutMatrix(size_t channel_out, size_t width_out, size_t height_out) {
  channel_out_ = channel_out;
  width_out_ = width_out;
  height_out_ = height_out;
}

void CudaConv::Launch(const float *in_matrix, const float *kernel, float *out_matrix) {
  float *d_kernel;
  size_t kernel_byte_size = channel_in_ * channel_out_ * kernel_size_ * kernel_size_ * sizeof(float);
  // allocate memory
  CHECK(cudaMalloc(&d_kernel, kernel_byte_size));
  // HtoD kernel
  CHECK(cudaMemcpy(d_kernel, kernel, kernel_byte_size, cudaMemcpyHostToDevice));

  float *d_in;
  float *d_out;
  size_t input_byte_size = width_in_ * height_in_ * channel_in_ * sizeof(float); 
  size_t output_byte_size = width_out_ * height_out_ * channel_out_ * sizeof(float); 

  // allocate memory
  CHECK(cudaMalloc(&d_in, input_byte_size));
  CHECK(cudaMalloc(&d_out, output_byte_size));

  // HtoD in_matrix
  CHECK(cudaMemcpy(d_in, in_matrix, input_byte_size, cudaMemcpyHostToDevice));

  // call kernel
  dim3 grid_size((width_out_ - 1) / BLOCK_SIZE.x + 1, (height_out_ - 1) / BLOCK_SIZE.y + 1);
  conv_kernel<<<grid_size, BLOCK_SIZE>>>(d_in, channel_in_, width_in_, height_out_,
                                         d_kernel, kernel_size_,
                                         d_out, channel_out_, width_out_, height_out_);

  // check kernel error
  cudaError_t errSync  = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

  // DtoH out_matrix
  CHECK(cudaMemcpy(out_matrix, d_out, output_byte_size, cudaMemcpyDeviceToHost));

  // free
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_kernel));
}
