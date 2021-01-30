import subprocess
from pathlib import Path
from typing import List, Union


class CppCompiler:
    """Compiles C++ code
    """
    def __init__(self, flags: List[str] = []):
        self._cmd_template = 'g++ -std=c++11 -O2 %s -o %%s %%s' % ' '.join(flags)

    def __call__(self, sources: Union[Path, List[Path]], out: Path) -> bool:
        """Compiles a list of C++ sources.

        Args:
            sources (List[FilePath]|FilePath): Source or list of sources.
            out (FilePath): Output path for binary

        Returns:
            bool: True if compilations succeed, False otherwise.
        """
        out_str = '"%s"' % out
        sources = [sources] if isinstance(sources, Path) else sources
        sources_str = ' '.join('"%s"' % w for w in sources)
        cmd = self._cmd_template % (out_str, sources_str)

        complete = subprocess.run(cmd,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  shell=True,
                                  check=False)
        return complete.returncode == 0


class JavaCompiler:
    """Compiles Java code
    """
    def __init__(self, flags: List[str] = []):
        self._cmd_template = 'javac %s %%s' % ' '.join(flags)

    def __call__(self, sources: Union[Path, List[Path]]) -> bool:
        """Compiles a list of Java sources.

        Args:
            sources (List[FilePath]|FilePath): Source or list of sources.
            out (FilePath): Output path for bytecode

        Returns:
            bool: True if compilation succeed, False otherwise.
        """
        sources = [sources] if isinstance(sources, Path) else sources
        sources_str = ' '.join('"%s"' % w for w in sources)
        cmd = self._cmd_template % sources_str

        complete = subprocess.run(cmd,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  shell=True,
                                  check=False)
        return complete.returncode == 0


class LatexCompiler:
    """Compiles latex source"""
    def __init__(self,
                 cmd: str = 'pdflatex',
                 flags: List[str] = ['--shell-escape', '-interaction=batchmode']):
        """
        Args:
            cmd (str): command to compile files. default to pdflatex
            flags (List[str]): list of flags to pass to command
        """
        self._cmd = cmd
        self._flags = flags

    def __call__(self, source: Path) -> bool:
        """It compiles a latex source leaving the pdf in the same directory of
        the source.
        Args:
            source (FilePath): path of file to compile
        """
        flags = ' '.join(self._flags)
        cmd = 'cd "%s" && %s %s "%s"' % (source.parent, self._cmd, flags, source.name)
        print(cmd)
        complete = subprocess.run(cmd,
                                  shell=True,
                                  stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  check=False)
        return complete.returncode == 0
