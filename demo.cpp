/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
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
#include "lenet5.h"

int main() {
  auto lenet5 = Lenet5();

  Matrix random_matrix = Matrix::Random(28 * 28, 1); 
  std::cerr << random_matrix.cols() << " " << random_matrix.rows() << '\n';
  lenet5.forward(random_matrix);
  
  lenet5.save("lenet5_weight");

  return 0;
}
