import fnmatch
import os

from setuptools import setup

from ocimatic import __version__


def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results


def get_resources(package):
    curr_path = os.getcwd()
    os.chdir(package)
    resources = recursive_glob('resources', '*')
    os.chdir(curr_path)
    return resources


setup(
    name='ocimatic',
    version=__version__,
    author='Nico Lehmann',
    description='Automatize task creation for OCI',
    url='https://github.com/OCIoficial/ocimatic',
    long_description=(open('LICENSE.rst').read()),
    license='Beer-Ware',
    packages=['ocimatic'],
    entry_points={
        'console_scripts': [
            'ocimatic=ocimatic.main:main',
        ],
    },
    package_data={'ocimatic': get_resources('ocimatic')},
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    install_requires=["ansi2html>=1.4.2", "colorama>=0.3.9", "flask>=1.0.2"],
    extras_require={'dev': ['yapf', 'pylint-yapf']})
