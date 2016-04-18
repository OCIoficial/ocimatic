import glob
import os
import shutil
from functools import total_ordering
from tempfile import mkstemp, mkdtemp

from ocimatic import ui

def change_directory():
    """Changes directory to the contest root and returns the absolute path of the
    last directory before reaching the root, this correspond to the directory
    of the problem in which ocimatic was called. If the function reach system
    root the program exists.

    Returns:
        (Directory) If ocimatic was called inside a subdirectory of a task this
            function returns the the problem directory. Otherwise it returns None.
    """
    last_dir = None
    while not os.path.exists('.ocimatic_contest'):
        last_dir = os.getcwd()
        head, tail = last_dir, None
        while not tail:
            head, tail = os.path.split(head)
        os.chdir('..')
        if os.getcwd() == '/':
            ui.fatal_error('ocimatic was not called inside a contest.')
    task_call = None if not last_dir else Directory(last_dir)
    return (Directory(os.getcwd()), task_call)

@total_ordering
class FilePath(object):
    """Represents a path to a file. The file may not exists in the
    file system, however the directory where the file resides must exist.
    """
    def __init__(self, arg1, arg2=None):
        """
        Args:

            arg1 (Directory or str): If arg2 is present correspond to
                directory where the file resides. Otherwise it contains
                a full path to file.
            arg2 (str): Basename of the file (including extension).
        """
        if arg2:
            self._directory = arg1
            self._filename = arg2
        else:
            dirname = os.path.dirname(arg1)
            assert(os.path.exists(dirname))
            self._directory = Directory(dirname)
            self._filename = os.path.basename(arg1)

    def __str__(self):
        return self.path

    def open(self, mode):
        return open(self.path, mode)

    def copy(self, dest):
        shutil.copy2(self.path, dest.path)

    @staticmethod
    def tmpfile():
        tmp_path = mkstemp()[1]
        return FilePath(tmp_path)

    def remove(self):
        os.remove(self.path)

    @property
    def name(self):
        return self._filename

    @property
    def ext(self):
        """str: File extension with beginning dot. """
        return self.splitext()[1]

    @property
    def rootname(self):
        """str: Filename without extension."""
        return self.splitext()[0]

    @property
    def path(self):
        """str: Full path to file"""
        return os.path.join(self._directory.path, self._filename)

    def __eq__(self, other):
        return self.fullath == other.path

    def __lt__(self, other):
        return self.path < other.path

    def chext(self, ext):
        """Change extension of file

        Args:
            ext (string): new extension including dot.

        Returns:
            FilePath: file path with the new extension
        """
        return FilePath(self._directory, self.rootname+ext)

    def splitext(self):
        """Split filename in root name and extension.

        Returns:
            (str, str): A pair where the first component correspond
                to root name and the second to the extension.
        """
        return os.path.splitext(self.path)

    def mtime(self):
        """Returns modification time. If the file does not exists in
        the file system this functions returns the oldest possible date.

        Returns:
            float: Numbers of seconds since the epoch or -inf if the file
                does not exists.
        """
        if self.exists():
            return os.path.getmtime(self.path)
        else:
            return float('-Inf')

    def directory(self):
        """Returns the directory where the file resides.

        Returns:
            Directory
        """
        return self._directory

    def exists(self):
        """Returns true if the file actually exists in the file system

        Returns:
            bool
        """
        return os.path.exists(self.path)

    def __bool__(self):
        return self.exists()


@total_ordering
class Directory(object):
    """Represent a directory in the filesystem. The directory must always
    exists.
    """

    def __init__(self, path):
        """
        Args:
            path (str): Full path to directory.
        """
        assert(os.path.exists(path))
        self._path = os.path.abspath(path)

    def __str__(self):
        return self.path

    @staticmethod
    def tmpdir():
        return Directory(mkdtemp())

    def rmtree(self):
        shutil.rmtree(self.path)

    @property
    def basename(self):
        return os.path.basename(self._path)

    @property
    def path(self):
        """str: Full path to directory."""
        return self._path

    def chdir(self, path):
        """Changes directory

        Args:
            path (str)

        Returns:
            Directory:
        """
        return Directory(os.path.join(self.path, path))

    def __eq__(self, other):
        return self.fullath == other.path

    def __lt__(self, other):
        return self.path < other.path

    def lsfile(self, pattern='*'):
        """List files inside this directory. An optional glob pattern
        could be provided. This pattern is concatenated to the path
        directory.

        Returns:
            List[FilePath]
        """
        pattern = os.path.join(self.path, pattern)
        files = [FilePath(f) for f in glob.glob(pattern) if os.path.isfile(f)]
        return sorted(files)

    def lsdir(self):
        """List directories inside this directory

        Returns:
            List[Directory]
        """
        pattern = os.path.join(self.path, '*')
        dirs = [Directory(f) for f in glob.glob(pattern) if os.path.isdir(f)]
        return sorted(dirs)

    def find_file(self, filename):
        """Finds a file by name in this directory.

        Returns:
            Optional[FilePath]
        """
        files = self.lsfile(filename)
        return files[0] if len(files) > 0 else None
