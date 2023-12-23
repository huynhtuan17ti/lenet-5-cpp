#include <catch2/catch_test_macros.hpp>
#include "cnn/layer/conv.h"

TEST_CASE("Cuda conv layer", "[layer]") {
#ifndef USE_CUDA
  std::cerr << "Nothing to test, please use cuda mode for this!\n";
#else
  // Matrix weight, shape = (channel_in*h_kernel*w_kernel, channel_out)
  // Vector bias, size = channel_out

  size_t channel_in = 3, height_in = 28, width_in = 28;
  size_t height_kernel = 5, width_kernel = 5;
  size_t channel_out = 6;

  auto conv = Conv(channel_in, height_in, width_in, channel_out, height_kernel, width_kernel);

  Matrix weight = Matrix::Random(channel_in * height_kernel * width_kernel, channel_out);
  Vector bias = Vector::Random(channel_out);
  conv.set_weight(weight);
  conv.set_bias(bias);

  Matrix input = Matrix::Random(channel_in * height_in * width_in, 1);
  //{
  //auto tmp = input;
  //tmp.resize(28, 28);
  //Matrix x = tmp.block(0, 0, 5, 5);
  //float res = 0;
  //for(size_t i = 0; i < x.size(); ++i) {
  //std::cerr << weight(i, 0) << " " << x(i) << '\n';
  //res += x(i) * weight(i, 0);
  //}
  //res += bias(0);
  //std::cerr << res << '\n';
  //}

  conv.forward(input);
  Matrix cuda_output = conv.output();

  conv.non_cuda_forward(input);
  Matrix output = conv.output();

  auto diff = (cuda_output - output).sum();
  REQUIRE(abs(diff) < 0.001);
#endif
}
