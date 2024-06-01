#include <gtest/gtest.h>
#include "parser/mnists_parser.hpp"

TEST(mnists_parser, size_test){
  const string train_X_path = CMAKE_SOURCE_DIR "/dataset/train/train-images.idx3-ubyte";
  const string train_y_path = CMAKE_SOURCE_DIR "/dataset/train/train-labels.idx1-ubyte";
  const string test_X_path = CMAKE_SOURCE_DIR "/dataset/test/t10k-images.idx3-ubyte";
  const string test_y_path = CMAKE_SOURCE_DIR "/dataset/test/t10k-labels.idx1-ubyte";
  vector<vector<unsigned char> > train_X = getInputImgs(train_X_path);
  vector<int> train_y = getLabels(train_y_path);
  vector<vector<unsigned char> > test_X = getInputImgs(test_X_path);
  vector<int> test_y = getLabels(test_y_path);

  EXPECT_EQ(train_X.size(), train_y.size());
  EXPECT_EQ(test_X.size(), test_y.size());
  int idx = 4;
  for (int i = 0; i < 28 * 28; i++) {
    if ((i % 28) == 0) cout << "\n";
    cout << train_X[idx][i] << " ";
  }
  cout << "y : " << train_y[idx];

}

TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}
