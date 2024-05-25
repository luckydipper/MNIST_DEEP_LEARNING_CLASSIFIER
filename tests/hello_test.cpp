#include <gtest/gtest.h>

TEST(StringTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("world", "m");
  // Expect equality.
  EXPECT_EQ(7 * 6, 43);
}
