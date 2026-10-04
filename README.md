# Ocimatic

Ocimatic is a tool for automating the work related to the creation of tasks for the Chilean Olympiad in Informatics (OCI).

## Installation

1. With `pip`

   ```bash
   pip install git+https://github.com/OCIoficial/ocimatic
   ```

2. With `uv`

   ```bash
   uv tool install git+https://github.com/OCIoficial/ocimatic
   ```

## Usage

To get started, run the following command to display a summary of available subcommands and options:

```bash
ocimatic -h
```

Ocimatic is designed to be discoverable. When you initialize a task, it will include sample files
that demonstrate various features of Ocimatic. Many directories also contain `README.md` files
documenting specific functionality. We recommend reading these README files and the comments in
the sample files to learn how to use Ocimatic.

## Editor support

There are VS Code and Zed extensions that provide syntax highlighting and language support for
`testplan.txt` files. They need `ocimatic` in your `PATH` to run the language server.

- **VS Code**: install [ocimatic-testplan](https://marketplace.visualstudio.com/items?itemName=nlehmann.ocimatic-testplan)
  from the Marketplace.
- **Zed**: clone [ocimatic-zed](https://github.com/OCIoficial/ocimatic-zed) and install it as a
  [dev extension](https://zed.dev/docs/extensions/developing-extensions#developing-an-extension-locally).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setting up a development environment, running tests,
and releasing.
