
/*
 * Evaluate model with Fashion MNIST test set
 * */
#include <Eigen/Dense>

#include "cnn/loss.h"
#include "cnn/loss/cross_entropy_loss.h"
#include "cnn/mnist.h"
#include "cnn/network.h"
#include "cnn/optimizer/sgd.h"
#include "cnn/timer.h"
#include "cnn/utils.h"
#include "include/lenet5.h"

int main() {
  Network lenet5 = Lenet5();

  // data
  MNIST dataset("data/fashion_mnist/");
  dataset.read();

  lenet5.load("weights/lenet5_mnist_weight");

  lenet5.timer_forward(dataset.test_data);
  float acc = compute_accuracy(lenet5.output(), dataset.test_labels);
  std::cout << "Test acc = " << acc << std::endl;
}
