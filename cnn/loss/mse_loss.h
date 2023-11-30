#pragma once

#include "cnn/loss.h"

class MSE : public Loss {
 public:
  void evaluate(const Matrix& pred, const Matrix& target);
};
