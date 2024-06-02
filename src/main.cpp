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
  // virtual Eigen::MatrixXd backward(void) = 0;
};

// bias term은 default로 존재.
struct Linear : public ModelInterface {
  const int input_size, output_size;
  Eigen::MatrixXd weight, delta_weight;
  Eigen::VectorXd bias, delta_bias;
  Eigen::MatrixXd input_data;
  explicit Linear(const int input_size, const int output_size)
      : input_size(input_size), output_size(output_size),
        weight(createHeWeight(input_size, output_size)),
        bias(Eigen::VectorXd::Zero(output_size)) {
    ;
  };
  ~Linear() = default;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &X) override {
    assert(X.cols() == weight.rows() && "forward errror.");
    cout << "X: \n"
         << X << "\nweight: " << weight << "\nbias: " << bias << "\n result: ";
    input_data = X;
    Eigen::MatrixXd zero_passing = X * weight;
    zero_passing.rowwise() += bias.transpose();
    return zero_passing;
  }
};

// struct FullConClassifyModel{

//   FullConClassifyModel(std::initializer_list<> s){;};
// };

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

  // MatrixXd a = convertToEigenMatrix(train_X[0],28*28,1);
  // l1.forward(a);

  neural::Linear lnas{4 * 4, 4 * 4};
  MatrixXd bac = MatrixXd::Ones(5, 16);
  MatrixXd result = lnas.forward(bac);
  // Softmax softmax_l(in_out_size[2], in_out_size[3]);

  // Model fcn{l1, l2, l3, softmax_l};
  // y_hat = fcn(X);
  // Loss loss(gt, y_hat) // predicted 결과 나옴 forwawad -> computation graph
  // 만듬

  //     vector<int>
  //         aa = {2, 31, 4, 1, 53, 2, 14, 1, 1, 2, 23, 4, 23};
  // const int num_batch = 20;

  // // DataLoader num_batch
  // for (int i = 0; i < aa.size(); i += num_batch) {
  //   // eigen 으로 변환
  // }
}
