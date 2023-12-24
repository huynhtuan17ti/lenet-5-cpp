/*
 * CNN demo for MNIST dataset
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "cnn/layer.h"
#include "cnn/layer/conv.h"
#include "cnn/layer/fully_connected.h"
#include "cnn/layer/max_pooling.h"
#include "cnn/layer/relu.h"
#include "cnn/layer/softmax.h"
#include "cnn/loss.h"
#include "cnn/loss/cross_entropy_loss.h"
#include "cnn/network.h"
#include "cnn/optimizer/sgd.h"
#include "include/lenet5.h"

int main() {
  auto lenet5 = Lenet5();
  Matrix random_matrix = Matrix::Random(28 * 28, 1);
  lenet5.forward(random_matrix);
  return 0;
}
