#include "cuda_conv.h"
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

CudaConv::~CudaConv() {
  CHECK(cudaFree(d_kernel_));
}

__global__ void conv_kernel_v1(float *in_matrix, size_t in_channel, size_t in_width, size_t in_height, 
                            float *kernel, int kernel_size, 
                            float *out_matrix, size_t out_channel, size_t out_width, size_t out_height) {
	size_t out_row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t out_col = blockIdx.x * blockDim.x + threadIdx.x;

  const size_t INPUT_HW = in_width * in_height;
  const size_t KERNEL_HW = kernel_size * kernel_size;
  const size_t KERNEL_HWC = in_channel * KERNEL_HW;
  const size_t OUTPUT_HW = out_width * out_height;

	if (out_row < out_height && out_col < out_width) {
		size_t out_id = out_row * out_width + out_col;
    for(size_t out_channel_id = 0; out_channel_id < out_channel; ++out_channel_id) {
      float res = 0;
      for(size_t in_row = out_row; in_row < out_row + kernel_size; ++in_row) {
        for(size_t in_col = out_col; in_col < out_col + kernel_size; ++in_col) {
          size_t kernel_row = in_row - out_row, kernel_col = in_col - out_col;
          size_t kernel_id = kernel_row * kernel_size + kernel_col;

          size_t in_id = in_row * in_width + in_col; 
          for(int in_channel_id = 0; in_channel_id < in_channel; ++in_channel_id) {
            size_t actual_kernel_id = out_channel_id * KERNEL_HWC + (in_channel_id * KERNEL_HW + kernel_id);
            float kernel_val = kernel[actual_kernel_id];
            res += kernel_val * in_matrix[in_channel_id * INPUT_HW + in_id];
          }
        }
      }
      out_matrix[out_channel_id * OUTPUT_HW + out_id] = res;
    }
	}
}

__global__ void conv_kernel_v2(float *in_matrix, size_t in_channel, size_t in_width, size_t in_height, 
                            float *kernel, int kernel_size, 
                            float *out_matrix, size_t out_channel, size_t out_width, size_t out_height) {

  const size_t INPUT_HW = in_width * in_height;
  const size_t KERNEL_HW = kernel_size * kernel_size;
  const size_t KERNEL_HWC = in_channel * KERNEL_HW;
  const size_t OUTPUT_HW = out_width * out_height;

  extern __shared__ float s_in_matrix[]; 
  size_t s_width = blockDim.x + kernel_size - 1;
  size_t s_height = blockDim.y + kernel_size - 1;
  const size_t SHARED_HW = s_width * s_height;
  size_t kernel_radius = kernel_size / 2;

  size_t out_r = blockIdx.y * blockDim.y + threadIdx.y;
	size_t out_c = blockIdx.x * blockDim.x + threadIdx.x;
  
  size_t visual_out_r = out_r + kernel_radius;
  size_t visual_out_c = out_c + kernel_radius;

  // index r and c in shared block visualization
  size_t block_r = threadIdx.y + kernel_radius;
  size_t block_c = threadIdx.x + kernel_radius;

  for(size_t channel = 0; channel < in_channel; ++channel) {
    for(int dr = -1; dr <= 1; ++dr)
      for(int dc = -1; dc <= 1; ++dc) {
        size_t inp_r = visual_out_r + dr * kernel_radius;
        size_t inp_c = visual_out_c + dc * kernel_radius;

        size_t inp_block_r = block_r + dr * kernel_radius; 
        size_t inp_block_c = block_c + dc * kernel_radius; 
      
        size_t _inp_id = channel * INPUT_HW + (inp_r * in_width + inp_c); 
        size_t _s_id = channel * SHARED_HW + (inp_block_r * s_width + inp_block_c); 
        s_in_matrix[_s_id] = in_matrix[_inp_id];
    }
  }

  __syncthreads();

  if (out_r < out_height && out_c < out_width) {
    for(size_t out_channel_id = 0; out_channel_id < out_channel; ++out_channel_id) {
      float res = 0;
      for(size_t in_channel_id = 0; in_channel_id < in_channel; ++in_channel_id) {
        for(size_t i = 0; i < kernel_size; ++i)
          for(size_t j = 0; j < kernel_size; ++j) {
            size_t _tmp1 = in_channel_id * KERNEL_HW + (i * kernel_size + j);           
            size_t kernel_id = out_channel_id * KERNEL_HWC + _tmp1;
            _tmp1 = (block_r + i - kernel_radius) * s_width + (block_c + j - kernel_radius);
            size_t shared_id = in_channel_id * SHARED_HW + _tmp1;

            res += kernel[kernel_id] * s_in_matrix[shared_id]; 
          }
      }
      size_t _out_id = out_r * out_width + out_c;
      out_matrix[out_channel_id * OUTPUT_HW + _out_id] = res;
    }
  }
}
void CudaConv::InitKernelParams(float* kernel) {
  size_t kernel_byte_size = channel_in_ * channel_out_ * kernel_size_ * kernel_size_ * sizeof(float);
  // allocate memory
  CHECK(cudaMalloc(&d_kernel_, kernel_byte_size));
  // HtoD kernel
  CHECK(cudaMemcpy(d_kernel_, kernel, kernel_byte_size, cudaMemcpyHostToDevice));
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

void CudaConv::Launch(const float *in_matrix, float *out_matrix) {
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
  //conv_kernel_v1<<<grid_size, BLOCK_SIZE>>>(d_in, channel_in_, width_in_, height_in_,
                                         //d_kernel_, kernel_size_,
                                         //d_out, channel_out_, width_out_, height_out_);

  size_t shared_size = channel_in_ * (kernel_size_ - 1 + BLOCK_SIZE.x) * 
                       (kernel_size_ - 1 + BLOCK_SIZE.y) * sizeof(float);
  conv_kernel_v2<<<grid_size, BLOCK_SIZE, shared_size>>>(d_in, channel_in_, width_in_, height_in_,
                                         d_kernel_, kernel_size_,
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
}
