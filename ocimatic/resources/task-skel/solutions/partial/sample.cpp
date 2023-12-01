// This file is a partial solution for the sample task. You can add partial
// solutions just by placing them in this directory. Remove this file before
// you start working on the actual task.
//
// Partial solutions must specify the list of subtasks they should fail with a
// comment as exemplified below. Ocimatic will check the solution fails these
// subtasks and only these subtasks. If no comment is present, ocimatic will
// assume all subtasks should fail.
//
// For the sample task, this solution should fail the subtask 2 because it uses
// `int` instead of `long`. This implies the solution should pass subtask 1.

// @ocimatic should-fail=[st2]
#include <iostream>

int main() {
  int32_t a, b;
  std::cin >> a >> b;

  std::cout << a + b << std::endl;

  return 0;
}
