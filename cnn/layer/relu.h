#pragma once

#include "cnn/layer.h"

class ReLU : public Layer {
 public:
  void forward(const Matrix& bottom);
  void backward(const Matrix& bottom, const Matrix& grad_top);
};
