import subprocess

from ocimatic.filesystem import FilePath


class CppCompiler:
    """Compiles C++ code
    """

    def __init__(self, flags=('-std=c++11', '-O2')):
        self._cmd_template = 'g++ %s -o %%s %%s' % ' '.join(flags)

    def __call__(self, sources, out):
        """Compiles a list of C++ sources.

        Args:
            sources (List[FilePath]|FilePath): Source or list of sources.
            out (FilePath): Output path for binary

        Returns:
            bool: True if compilations succeed, False otherwise.
        """
        out = '"%s"' % out
        sources = [sources] if isinstance(sources, FilePath) else sources
        sources = ' '.join('"%s"' % w for w in sources)
        cmd = self._cmd_template % (out, sources)

        complete = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        return complete.returncode == 0


class JavaCompiler:
    """Compiles Java code
    """

    def __init__(self, flags=()):
        self._cmd_template = 'javac %s %%s' % ' '.join(flags)

    def __call__(self, sources):
        """Compiles a list of Java sources.

        Args:
            sources (List[FilePath]|FilePath): Source or list of sources.
            out (FilePath): Output path for bytecode

        Returns:
            bool: True if compilation succeed, False otherwise.
        """
        sources = [sources] if isinstance(sources, FilePath) else sources
        sources = ' '.join('"%s"' % w for w in sources)
        cmd = self._cmd_template % sources

        complete = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        return complete.returncode == 0


class LatexCompiler:
    """Compiles latex source"""

    def __init__(self, cmd='pdflatex', flags=('--shell-escape', '-interaction=batchmode')):
        """
        Args:
            cmd (str): command to compile files. default to pdflatex
            flags (List[str]): list of flags to pass to command
        """
        self._cmd = cmd
        self._flags = flags

    def __call__(self, source):
        """It compiles a latex source leaving the pdf in the same directory of
        the source.
        Args:
            source (FilePath): path of file to compile
        """
        flags = ' '.join(self._flags)
        cmd = 'cd "%s" && %s %s "%s"' % (source.directory(), self._cmd, flags, source.name)
        complete = subprocess.run(
            cmd,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        return complete.returncode == 0
