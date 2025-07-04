// This file is a partial solution for the sample task. You can add partial
// solutions just by placing them in this directory. Remove this file before
// you start working on the actual task.
//
// Each partial solution must specify its expected outcome for each subtask
// using a comment as exemplified below. Each subtask must be marked with one of
// the possible outcomes:
// - 'OK': The solution passes all the test cases in the subtask.
// - 'FAIL': The solution fails at least one test case for any reason.
// - 'TLE': The solution exceeds the time limit in at least one test case and
//    no test case fails for a reason other than the time limit being exceeded.
// - 'WA': The solution produces a wrong answer in at least one test case
//   and no test case fails for a reason other than giving a wrong answer.
//
// For the sample task, this solution passes all test cases in subtask 1 but it
// should give a wrong answer in subtask 2 because it uses `int` instead of
// `long`.

// @ocimatic::expected [st1=OK, st2=WA]
#include <iostream>

int main() {
  int32_t a, b;
  std::cin >> a >> b;

  std::cout << a + b << std::endl;

  return 0;
}
