#include <gtest/gtest.h>

TEST(StringTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("world", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}
