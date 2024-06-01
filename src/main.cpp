#include "parser/mnists_parser.hpp"
#include <iostream>
using namespace std;

int main() {
  cin.tie(nullptr);
  ios_base::sync_with_stdio(false);
  cout << "hello world";

  const string train_X_path = "../dataset/train/train-images.idx3-ubyte";
  const string train_y_path = "../dataset/train/train-labels.idx1-ubyte";
  const string test_X_path = "../dataset/test/t10k-images.idx3-ubyte";
  const string test_y_path = "../dataset/test/t10k-labels.idx1-ubyte";
  vector<vector<unsigned char>> train_X = getInputImgs(train_X_path);
  vector<int> train_y = getLabels(train_y_path);
  vector<vector<unsigned char>> test_X = getInputImgs(test_X_path);
  vector<int> test_y = getLabels(test_y_path);

}
