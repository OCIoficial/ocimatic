// This file is a partial solution for the sample task. You can add partial
// solutions just by placing them in this directory. Remove this file before
// you start working on the actual task.
//
// Partial solutions must specify which subtasks they pass or fail. You can use
// either a should-pass or a should-faill comment for this purpose, specifying
// the list of subtasks the solution should pass or fail. If no comment is
// present, ocimatic will assume all subtasks should fail. For the sample task,
// this solution should fail the second subtask because it uses int instead of
// long.

// @ocimatic should-fail=[st2]
#include <iostream>

int main() {
  int32_t a, b;
  std::cin >> a >> b;

  std::cout << a + b << std::endl;

  return 0;
}
