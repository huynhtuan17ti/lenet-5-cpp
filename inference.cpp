
/*
 * Train model with Fashion MNIST
 * */
#include <Eigen/Dense>

#include "cnn/loss.h"
#include "cnn/loss/cross_entropy_loss.h"
#include "cnn/mnist.h"
#include "cnn/network.h"
#include "cnn/optimizer/sgd.h"
#include "cnn/utils.h"
#include "lenet5.h"
#include "timer.h"

int main() {
  Network lenet5 = Lenet5();

  // data
  MNIST dataset("data/fashion_mnist/");
  dataset.read();

  lenet5.load("weights/lenet5_mnist_weight");

  Timer clock;
  clock.tick();
  lenet5.forward(dataset.test_data);
  clock.tock();
  float acc = compute_accuracy(lenet5.output(), dataset.test_labels);
  std::cout << "Run time = " << clock.duration().count() << " ms\n";
  std::cout << "Test acc = " << acc << std::endl;
}
