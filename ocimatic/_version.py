from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ocimatic")
except PackageNotFoundError:
    __version__ = "not found"
