import fnmatch
import os

from setuptools import setup

__version__ = "0.1.0"


pkg_dir = os.path.dirname(os.path.realpath(__file__))


def recursive_glob(treeroot: str, pattern: str) -> list[str]:
    results: list[str] = []
    for base, dirs, files in os.walk(treeroot):
        del dirs
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results


def get_resources(package: str) -> list[str]:
    curr_path = os.getcwd()
    os.chdir(package)
    resources = recursive_glob("resources", "*") + recursive_glob("templates", "*")
    os.chdir(curr_path)
    return resources


def get_requires(filepath: str) -> list[str]:
    if os.path.isfile(filepath):
        with open(filepath) as f:
            return f.read().splitlines()
    return []


setup(
    name="ocimatic",
    version=__version__,
    author="Nico Lehmann",
    description="Automatize task creation for OCI",
    url="https://github.com/OCIoficial/ocimatic",
    long_description=(open("LICENSE.rst").read()),  # noqa: SIM115
    license="Beer-Ware",
    packages=["ocimatic"],
    entry_points={
        "console_scripts": [
            "ocimatic=ocimatic.main:main",
        ],
    },
    package_data={"ocimatic": get_resources("ocimatic")},
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    install_requires=get_requires(os.path.join(pkg_dir, "requirements.txt")),
    extras_require={"dev": get_requires(os.path.join(pkg_dir, "requirements-dev.txt"))},
)
