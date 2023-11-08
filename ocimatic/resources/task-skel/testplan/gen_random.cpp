// This script generates random test cases for the sample task. It takes a
// minimum and maximum value and prints two random numbers in that range.
// See the comment in testplan.txt for further details.

#include <cassert>
#include <iostream>
#include <random>

int main(int argc, char *argv[]) {
  //  The argument in position 1 is the hidden seed.
  std::hash<std::string> hasher;
  std::mt19937 gen(hasher(argv[1]));

  // The min and max are in positions 2 and 3
  const int MIN = std::atoi(argv[2]);
  const int MAX = std::atoi(argv[3]);

  // Check that the range is valid
  assert(MIN <= MAX);

  std::uniform_int_distribution<int> dist(MIN, MAX);
  std::cout << dist(gen) << " " << dist(gen) << std::endl;

  return 0;
}
