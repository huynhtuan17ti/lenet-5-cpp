#pragma once

#include "cnn/layer.h"
#include "cnn/layer/conv.h"
#include "cnn/layer/fully_connected.h"
#include "cnn/layer/max_pooling.h"
#include "cnn/layer/relu.h"
#include "cnn/layer/softmax.h"
#include "cnn/network.h"

inline Network Lenet5() {
  Layer* conv1 = new Conv(1, 28, 28, 6, 5, 5);
  Layer* relu1 = new ReLU;
  // output shape: (24, 24, 6)
  Layer* pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
  Layer* conv2 = new Conv(6, 12, 12, 16, 5, 5);
  Layer* relu2 = new ReLU;
  Layer* pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
  Layer* dense1 = new FullyConnected(pool2->output_dim(), 120);
  Layer* relu3 = new ReLU;
  Layer* dense2 = new FullyConnected(120, 84);
  Layer* relu4 = new ReLU;
  Layer* dense3 = new FullyConnected(84, 10);
  Layer* softmax = new Softmax;

  Network lenet5;
  lenet5.add_layer(conv1);
  lenet5.add_layer(relu1);
  lenet5.add_layer(pool1);
  lenet5.add_layer(conv2);
  lenet5.add_layer(relu2);
  lenet5.add_layer(pool2);
  lenet5.add_layer(dense1);
  lenet5.add_layer(relu3);
  lenet5.add_layer(dense2);
  lenet5.add_layer(relu4);
  lenet5.add_layer(dense3);
  lenet5.add_layer(softmax);

  return lenet5;
}
