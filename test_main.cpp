#include "gtest/gtest.h"

int my_argc;
char** my_argv;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  my_argc= argc;
  my_argv= argv;
  for (int i = 0; i < my_argc; i++) {
        std::cout << i << ":" << my_argv[i] << std::endl;
  }
  return RUN_ALL_TESTS();
}
