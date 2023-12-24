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
#include "include/lenet5.h"

int main() {
  Network lenet5 = Lenet5();

  // data
  MNIST dataset("data/fashion_mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;

  // loss
  Loss* loss = new CrossEntropy;
  lenet5.add_loss(loss);
  // train & test
  SGD opt(0.001, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 5;
  const int batch_size = 128;

  for (int epoch = 0; epoch < n_epoch; epoch++) {
    shuffle_data(dataset.train_data, dataset.train_labels);
    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
      int ith_batch = start_idx / batch_size;
      Matrix x_batch =
          dataset.train_data.block(0, start_idx, dim_in, std::min(batch_size, n_train - start_idx));
      Matrix label_batch =
          dataset.train_labels.block(0, start_idx, 1, std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);
      if (false && ith_batch % 10 == 1) {
        std::cout << ith_batch << "-th grad: " << std::endl;
        lenet5.check_gradient(x_batch, target_batch, 10);
      }
      lenet5.forward(x_batch);
      lenet5.backward(x_batch, target_batch);
      // display
      if (ith_batch % 50 == 0) {
        std::cout << ith_batch << "-th batch, loss: " << lenet5.get_loss() << std::endl;
      }
      // optimize
      lenet5.update(opt);
    }
    // test
    lenet5.forward(dataset.test_data);
    float acc = compute_accuracy(lenet5.output(), dataset.test_labels);
    std::cout << std::endl;
    std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
    std::cout << std::endl;
  }

  lenet5.save("lenet5_mnist_weight");
}
