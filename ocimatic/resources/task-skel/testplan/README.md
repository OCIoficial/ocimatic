# Test Plan

A test plan is a file describing how to generate test cases for a task. The purpose of a test plan is twofold. First, it's a convenient way to abstract over the specifics of where to place generated files, so you can focus on generating the content. Second, it should serve as documentation of how tests are generated and what they are supposed to be testing. Consider that other people *will* read the test plan and may need to tweak it or fix small bugs. We encourage you to use comments.

## Structure

A test plan is divided into subtasks. A subtask is declared with a header `[Subtask N]`, where `N` is the number of the subtask (starting from 1). Subtasks must be declared in order.

Inside a subtask, each line specifies a command to generate one or multiple test cases. A command has the form `group ; cmd ...args`, where `group` is an identifier used to group related tests, `cmd` is one of the possible commands, and `...args` are the arguments to the command. The group will be used as part of the name of the generated file, and it's useful when debugging solutions to be able to see at a glance what type of cases the solution is failing. Try to use a group name that describes how the test was generated or what it's supposed to be testing, for example, you can use `rand-large` for random tests with large parameters.

## Commands

A command can be either `copy`, `echo`, or a file containing a generator script. Check the sample `testplan.txt` next to this README to see how each command is used.

* `copy`:
  The copy command takes as a single argument a glob pattern. The command will copy all files matching the pattern, relative to the root of the current task.
* `echo`:
  This command takes one or more arguments and prints them in a single line. This can be useful to quickly specify manual test cases for tasks where the input consists of a single line.
* `script`:
  A generator script is a file in either Python (extension `.py`) or C++ (extension `.cpp`). The file should be placed next to `testplan.txt`. When processing the test plan, `ocimatic` will run the generator with the provided arguments (`sys.argv` or `**argv`). The generator should then write to the standard output to produce the test case.

  A script must be deterministic and generate the same result every time it's executed. This is in conflict with a generator wanting to generate *arbitrary* values. Since the script must be deterministic, true randomness cannot be used. To this end, `ocimatic` passes an additional (hidden) argument to each invocation, which is guaranteed to be different for each invocation. The generator can use this extra argument to seed the random generator. The extra argument is passed as the first argument, meaning that the rest of the arguments are *shifted* by one position.

## Input Validators

An input validator is a script that checks whether the input of a test case satisfies the format and restrictions in the statement. You can specify a validator for a subtask in the subtask's header. See `testplan.txt` for an example. Validators are optional, but their use is highly encouraged.

## The `@extends` directive

It's common for subtasks to be cumulative, i.e., a solution for a harder subtask subsumes and should solve all test cases for an easier subtask. The `@extends` directive can be used to specify that a subtask extends from another, and should include all its test cases. The format is `@extends subtask N`, where `N` is the number of the subtask extending from.

If a subtask `N` extends from `M`, we say `M` is a parent of `N`. The extends relationship is transitive. We call ancestors of `N` to all subtasks reachable from `N` following the extends relationship.

The `@extends` directive *does not duplicate* test cases. Solutions still run once per test case, but the extra information is used for validation. For example, input validators are run on the target subtask and all its ancestors. Similarly, when validating if partial solutions pass/fail the appropriate subtasks, the extends relationship is also considered.

## Multi-test Script

Normally, a script generates a single test case. Then, it can be used multiple times in the test plan. This is convenient because it lets you focus on the properties of a single test when writing the script and defer to the test plan how to use it coherently to form the dataset. However, sometimes it is convenient to generate a set of test cases *programmatically*. For example, if a subtask has a finite set of possible cases (common for easy subtasks), you may want to include them all. To accommodate this, a script may also generate multiple test cases in a single invocation.

To generate multiple test cases, the script must write the [file separator control code](https://en.wikipedia.org/wiki/C0_and_C1_control_codes#Field_separators) to signal the end of a test case. The file separator control code has a decimal representation of `28`. See `multi.py` as an example of a multi-test script.

Note that a multi-test script subsumes all other commands, as all the tests for a subtask could be generated by calling a single multi-test script. We advise using this feature judiciously. A test plan using multiple small scripts will typically be more readable than a huge monolithic script.
