#pragma once

#include "cnn/loss.h"

class CrossEntropy : public Loss {
 public:
  void evaluate(const Matrix& pred, const Matrix& target);
};
