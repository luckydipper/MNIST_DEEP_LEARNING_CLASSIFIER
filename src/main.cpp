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
    delta_bias =  delta_out.colwise().sum(); // 이게 왜 colwise
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
//
Eigen::MatrixXd softmaxRowwise(const Eigen::MatrixXd &mat) {
  Eigen::MatrixXd result(mat.rows(), mat.cols());

  // 각 행에 대해 소프트맥스 적용
  for (int i = 0; i < mat.rows(); ++i) {
    // 각 행의 최대값을 빼서 수치적으로 안정화
    double maxCoeff = mat.row(i).maxCoeff();
    Eigen::VectorXd expRow = (mat.row(i).array() - maxCoeff).exp();
    result.row(i) = expRow / expRow.sum();
  }

  return result;
}
double crossEntropyLoss(const Eigen::MatrixXd &predictions,
                        const Eigen::MatrixXd &groundTruth) {
  Eigen::MatrixXd logPredictions = predictions.array().log();
  double loss = -(groundTruth.array() * logPredictions.array()).sum();
  return loss;
}
// pure virtual을 쓰면 forward의 ouput이 double 이 될 수 가 없네?
struct SoftmaxLoss {
  SoftmaxLoss() = default;
  Eigen::MatrixXd ground_truth, predict;
  double loss;
  int batch_size;
  double forward(const Eigen::MatrixXd &X, const Eigen::MatrixXd &gt) {
    assert(X.rows() == gt.rows() && X.cols() == gt.cols());
    batch_size = X.rows();
    Eigen::MatrixXd y_hat = softmaxRowwise(X);
    ground_truth = gt, predict = y_hat;
    loss = crossEntropyLoss(y_hat, ground_truth);

    return loss;
  };

  Eigen::MatrixXd backward(double delta = 1) {
    return (predict-ground_truth).array() / batch_size; //predict에서 gt를 빼야함.
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

vector<MatrixXd> createImageBatches(const vector<vector<unsigned char>> &data,
                                    int batchSize) {
  vector<MatrixXd> batches;
  int numImages = data.size();
  int imageSize = 28 * 28; // 28x28 이미지를 플래튼한 크기

  // 이미지 크기와 배치 크기를 기반으로 행렬 생성
  for (int i = 0; i < numImages; i += batchSize) {
    int currentBatchSize = min(batchSize, numImages - i);
    MatrixXd batch(currentBatchSize, imageSize);

    for (int j = 0; j < currentBatchSize; ++j) {
      const vector<unsigned char> &image = data[i + j];
      for (int k = 0; k < imageSize; ++k) {
        batch(j, k) = static_cast<double>(image[k]) / 255.0; // 정규화
      }
    }

    batches.push_back(batch);
  }

  return batches;
}

vector<MatrixXd> createLabelBatches(const vector<int> &labels, int batchSize) {
  vector<MatrixXd> batches;
  int numLabels = labels.size();
  int numClasses = 10; // 0~9까지의 클래스 수

  // 레이블 크기와 배치 크기를 기반으로 행렬 생성
  for (int i = 0; i < numLabels; i += batchSize) {
    int currentBatchSize = min(batchSize, numLabels - i);
    MatrixXd batch = MatrixXd::Zero(currentBatchSize, numClasses);

    for (int j = 0; j < currentBatchSize; ++j) {
      int label = labels[i + j];
      batch(j, label) = 1.0; // 원핫 인코딩
    }

    batches.push_back(batch);
  }

  return batches;
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

  // model definition
  const int flatten_img_size = 28 * 28;
  const int in_out_size[] = {400, 300, 100, 10};
  neural::Linear l1{flatten_img_size, in_out_size[0]},
      l2{in_out_size[0], in_out_size[1]}, l3{in_out_size[1], in_out_size[2]},
      l4{in_out_size[2], in_out_size[3]};
  neural::ReLU act1{}, act2{}, act3{}, act4{};
  neural::SoftmaxLoss sft{};

  const int num_batch = 16;//, num_input = 16, num_out = 18;
  const double lr = 0.003;

  vector<MatrixXd> imgs_iter = createImageBatches(train_X, num_batch);
  vector<MatrixXd> lable_iter = createLabelBatches(train_y, num_batch);

  // python 의 ordered map으로 해결.
  for (int i = 0; i < imgs_iter.size(); i++) {
    auto l_1 = l1.forward(imgs_iter[i]);
    auto a1 = act1.forward(l_1);
    auto l_2 = l2.forward(a1);
    auto a2 = act2.forward(l_2);
    auto l_3 = l3.forward(a2);
    auto a3 = act3.forward(l_3);
    auto l_4 = l4.forward(a3);
    auto a4 = act4.forward(l_4);
    double current_loss = sft.forward(a4, lable_iter[i]);

    cout << i << " iter, current_loss is " << current_loss << "\n";
    auto b1 = sft.backward();
    auto b2 = act4.backward(b1);
    auto b3 = l4.backward(b2);
    auto b4 = act3.backward(b3);
    auto b5 = l3.backward(b4);
    auto b6 = act2.backward(b5);
    auto b7 = l2.backward(b6);
    auto b8 = act1.backward(b7);
    auto b9 = l1.backward(b8);
    vector<neural::Linear *> layers = {&l1, &l2, &l3, &l4};
    for (auto &layer : layers) {
      layer->weight.array() -= lr * layer->delta_weight.array();
      layer->bias.array() -= lr * layer->delta_bias.array();
    }
  }
}
