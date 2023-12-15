#include <catch2/catch_test_macros.hpp>
#include "cnn/layer/conv.h"

TEST_CASE("Cuda conv layer", "[layer]") {
#ifndef USE_CUDA
  std::cerr << "Nothing to test, please use cuda mode for this!\n";
#else
  // Matrix weight, shape = (channel_in*h_kernel*w_kernel, channel_out)
  // Vector bias, size = channel_out

  size_t channel_in = 1, height_in = 28, width_in = 28;
  size_t height_kernel = 5, width_kernel = 5;
  size_t channel_out = 6;

  auto conv = Conv(channel_in, height_in, width_in, channel_out, height_kernel, width_kernel);

  Matrix weight = Matrix::Random(channel_in * height_kernel * width_kernel, channel_out);
  Vector bias = Vector::Random(channel_out);
  conv.set_weight(weight);
  conv.set_bias(bias);

  Matrix input = Matrix::Random(channel_in * height_in * width_in, 1);
  conv.forward(input);
  Matrix cuda_output = conv.output();

  conv.non_cuda_forward(input);
  Matrix output = conv.output();

  auto diff = (cuda_output - output).sum();
  std::cerr << cuda_output.sum() << '\n';
  std::cerr << output.sum() << '\n';
  REQUIRE(abs(diff) < 0.001);
#endif
}
