Ocimatic
========

`ocimatic` is a tool for automating the work related to the creation of tasks for the Chilean Olympiad in Informatics (OCI).

Usage and Installation
------------------
```bash
$ git clone https://github.com/OCIoficial/ocimatic
$ cd ocimatic
$ ./bin/ocimatic
```

You can also add the `bin` directory to the `PATH`.

optional: install `colorama` to support colors (`pip install colorama`).

Basic Usage
----------
Assuming `ocimatic` is in the `PATH`

```bash
$ ocimatic -h
```

Run server
--------
First install `flask` (`pip install flask`). Then, assuming you are inside a contest:

```bash
$ ocimatic server start
```

This will start a server in port `9999`.

optional: install `colorama` and `ansi2html` to have color support (`pip install colorama ansi2html`)

