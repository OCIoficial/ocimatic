// This file is a partial solution for the sample task. Remove this file before
// you start working on the actual task. See the README.md in the solutions
// directory for more details on how to write solutions.
//
// For the sample task, this solution is expected to pass all test cases in
// subtask 1 but it should fail with a wrong answer in subtask 2 because it uses
// `int32_t` instead of `int64_t`.

// @ocimatic::expected [st1=OK, st2=WA]
#include <cstdint>
#include <iostream>

int main() {
  int32_t a, b;
  std::cin >> a >> b;

  std::cout << a + b << std::endl;

  return 0;
}
