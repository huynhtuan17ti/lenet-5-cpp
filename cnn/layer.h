#pragma once

#include <Eigen/Core>
#include <ostream>
#include <vector>
#include "optimizer.h"
#include "utils.h"

class Layer {
 protected:
  Matrix top;          // layer output
  Matrix grad_bottom;  // gradient w.r.t input

 public:
  virtual ~Layer() {}

  virtual void forward(const Matrix& bottom) = 0;
  virtual void backward(const Matrix& bottom, const Matrix& grad_top) = 0;
  virtual void update(Optimizer& opt) {}
  virtual const Matrix& output() { return top; }
  virtual const Matrix& back_gradient() { return grad_bottom; }
  virtual int output_dim() { return -1; }
  virtual std::vector<float> get_parameters() const { return std::vector<float>(); }
  virtual std::vector<float> get_derivatives() const { return std::vector<float>(); }
  virtual void set_parameters(const std::vector<float>& param) {}

  // read/write binary
  virtual void serialize(std::ostream& stream) const { return; }
  virtual void deserialize(std::istream& stream) { return; }
};
