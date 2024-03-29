#pragma once

#include <unordered_map>
#include "cnn/optimizer.h"

class SGD : public Optimizer {
 private:
  float momentum;                                  // momentum factor (default: 0)
  bool nesterov;                                   // enables Nesterov momentum (default: False)
  std::unordered_map<const float*, Vector> v_map;  // velocity

 public:
  explicit SGD(float lr = 0.01, float decay = 0.0, float momentum = 0.0, bool nesterov = false)
      : Optimizer(lr, decay), momentum(momentum), nesterov(nesterov) {}

  void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw);
};
