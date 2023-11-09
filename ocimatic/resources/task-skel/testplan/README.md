# Testplan

A testplan is a file describing how to generate test cases for a task. Each line specifies a command
to generate one or multiple test cases. When executing the testplan, ocimatic will take care of
choosing names for each generated file and placing them in the correct directory.

## Commands

A command has the form `<group> ; <cmd>`, where corresponds to a string describing the purpose
of the test and `<cmd>` describes which command to run.  The group is supposed to briefly describe
how the test was generated or what is supposed to be testing. The group will be used as part of the
name of the generated test file. This is useful when debugging solutions to be able to see at a
glance in which test cases the solution is failing. It should also inform a reader of the testplan
the purpose of each test.

A command could be either `copy` `echo` or a generator script.

`copy`: The copy command takes a single argument: a path to a file to copy. The path should be
relative to the root of the current task.

`echo`: This command takes one or more arguments and prints them in a single line. This can be
useful to quickly specify manual test cases for tasks where the input consist of a single line.

`script`: A generator script is a file in either Python (extension `.py`) or C++ (extension `.cpp`).
The file should be placed in `testplan/` next to this file. When processing the testplan, ocimatic
will run the script with the provided arguments (sys.argv or **argv). The script should then write
to the standard output.
Generator scripts typically use randomness. To ensure each execution of the testplan generates the
same results, a script should set the random seed to a fixed value. To this end, ocimatic passes
an additional (hidden) argument to each invocation which is guaranteed to be different for each
invocation. The generator should use this extra argument to generate the random seed. The extra
argument is passed as the first argument meaning that the rest of the arguments are "shifted" in
one position.
