Ocimatic
========

`ocimatic` is a tool for automating the work related to the creation of tasks for the Chilean Olympiad in Informatics (OCI).

Usage and Installation
------------------
```
$ git clone https://github.com/OCIoficial/ocimatic
$ cd ocimatic
$ ./bin/ocimatic
```

You can also add the `bin` directory to the `PATH`.

Basic Usage
----------
Assuming `ocimatic` is in the `PATH`

```
$ ocimatic contest new test_contest
$ cd test_contest
$ ocimatic new problem_a
$ cd problem_a
$ ocimatic run
$ ocimatic check
$ ocimatic pdf
```

