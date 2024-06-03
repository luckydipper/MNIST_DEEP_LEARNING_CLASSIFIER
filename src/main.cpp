#include "parser/mnists_parser.hpp"
#include <assert.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
using namespace std;

namespace neural {

// 이건 util로 가는 것이 맞는 듯
Eigen::MatrixXd createHeWeight(int rows, int cols) {
  double stddev = std::sqrt(2.0 / rows);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d(0, stddev);

  Eigen::MatrixXd matrix(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix(i, j) = d(gen);
    }
  }
  return matrix;
}

struct ModelInterface {
  ModelInterface() { ; };
  virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &X) = 0;
  virtual Eigen::MatrixXd backward(const Eigen::MatrixXd &delta_out) = 0;
};

// bias term은 default로 존재.
struct Linear : public ModelInterface {
  const int input_size, output_size;
  Eigen::MatrixXd weight, delta_weight;
  Eigen::VectorXd bias, delta_bias;
  Eigen::MatrixXd input_data, delta_input;
  explicit Linear(const int input_size, const int output_size)
      : input_size(input_size), output_size(output_size),
        weight(createHeWeight(input_size, output_size)),
        bias(Eigen::VectorXd::Zero(output_size)) {
    ;
  };
  ~Linear() = default;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &X) override {
    assert(X.cols() == weight.rows() && "forward errror.");
    input_data = X;

    Eigen::MatrixXd zero_passing = X * weight;
    zero_passing.rowwise() += bias.transpose();
    return zero_passing;
  }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &delta_out) override {
    delta_input = delta_out * weight.transpose();
    delta_bias = delta_out.rowwise().sum();
    delta_weight = input_data.transpose() * delta_out;
    return delta_input;
  }
};

struct ReLU : public ModelInterface {
  Eigen::MatrixXi mask;

  ReLU() = default;
  Eigen::MatrixXd forward(const Eigen::MatrixXd &X) override {
    Eigen::MatrixXd masked_input = X;
    mask = Eigen::MatrixXi::Ones(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); i++)
      for (int j = 0; j < X.cols(); j++)
        if (X(i, j) <= 0)
          masked_input(i, j) = mask(i, j) = 0;
    return masked_input;
  };

  Eigen::MatrixXd backward(const Eigen::MatrixXd &delta_out) override {
    Eigen::MatrixXd masked_out = delta_out;
    for (int i = 0; i < delta_out.rows(); ++i)
      for (int j = 0; j < delta_out.cols(); ++j)
        if (mask(i, j) == 0)
          masked_out(i, j) = 0.;
    return masked_out;
  };
};

} // namespace neural

using namespace Eigen;
MatrixXd convertToEigenMatrix(const vector<unsigned char> &data, int rows,
                              int cols) {
  assert(rows * cols == data.size());
  // Eigen 행렬 생성
  MatrixXd mat(rows, cols);

  // 데이터 복사
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      mat(i, j) = static_cast<double>(data[i * cols + j]);
    }
  }

  return mat;
}

int main() {
  cin.tie(nullptr);
  ios_base::sync_with_stdio(false);

  const string train_X_path = "../dataset/train/train-images.idx3-ubyte";
  const string train_y_path = "../dataset/train/train-labels.idx1-ubyte";
  const string test_X_path = "../dataset/test/t10k-images.idx3-ubyte";
  const string test_y_path = "../dataset/test/t10k-labels.idx1-ubyte";
  vector<vector<unsigned char>> train_X = getInputImgs(train_X_path);
  vector<int> train_y = getLabels(train_y_path);
  vector<vector<unsigned char>> test_X = getInputImgs(test_X_path);
  vector<int> test_y = getLabels(test_y_path);

  const int flatten_img_size = 28 * 28;
  const int in_out_size[] = {10 * 10, 5 * 5, 4 * 4, 10};
  neural::Linear l1{flatten_img_size, in_out_size[0]},
      l2{in_out_size[0], in_out_size[1]}, l3{in_out_size[1], in_out_size[2]};

  const int num_batch = 5, num_input = 16, num_out = 18;
  neural::Linear lnas{num_input, num_out};

  MatrixXd bac = MatrixXd::Ones(num_batch, num_input);
  MatrixXd result = lnas.forward(bac);
  lnas.backward(Eigen::MatrixXd::Ones(num_batch, 18));
  neural::ReLU rl1{};
  rl1.forward(result);
  cout << "\n\n" << result << "\n\n" << rl1.mask << "\n\n" << rl1.backward(result);
}
