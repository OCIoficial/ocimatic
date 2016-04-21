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
        (Directory) If ocimatic was called inside a subdirectory of
            corresponding to a task this function returns the the problem
            directory. Otherwise it returns None.
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
    """Represents a path to a file. The file may not exist in the
    file system, however the directory where the file resides must exist.
    """
    @staticmethod
    def tmpfile():
        tmp_path = mkstemp()[1]
        return FilePath(tmp_path)

    def __init__(self, arg1, arg2=None):
        """
        Args:
            arg1 (Directory|str): If arg2 is not None, it corresponds to
                the directory where the file resides. Otherwise it contains
                a full path to the file.
            arg2 (Optional[str]): name of the file (including extension).
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
        """Copy file in this path.
        Args:
           dest (FilePath): destination path
        """
        shutil.copy2(self.path, dest.path)

    def remove(self):
        """Removes file in this path from the filesystem"""
        os.remove(self.path)

    @property
    def name(self):
        """Name of the file.
        Returns:
           str
        """
        return self._filename

    @property
    def ext(self):
        """str: File extension beginning with dot. """
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
            (str, str): A pair where the first component corresponds
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
    exist.
    """
    @staticmethod
    def tmpdir():
        """Returns a temporary directory. The user is responsible
        of deleting the directory after using it.
        Returns:
            Directory
        """
        return Directory(mkdtemp())

    def __init__(self, path):
        """Produces an assertion error if the directory does not exist
        or if the path does not correspond to a directory.
        Args:
            path (str): Full path to directory.
        """
        assert(os.path.exists(path))
        assert(os.path.isdir(path))
        self._path = os.path.abspath(path)

    def __str__(self):
        return self.path

    def rmtree(self):
        """Removes this directory from the filesystem."""
        shutil.rmtree(self.path)

    @property
    def basename(self):
        """str: Name of this directory"""
        return os.path.basename(self._path)

    @property
    def path(self):
        """str: Full path to directory."""
        return self._path

    def chdir(self, path):
        """Changes directory. If the new directory does not exists
        this functions produces an assertion error.

        Args:
            path (str): A path relative to this directory.

        Returns:
            Directory: The new directory.
        """
        return Directory(os.path.join(self.path, path))

    def __eq__(self, other):
        return self.fullath == other.path

    def __lt__(self, other):
        return self.path < other.path

    def lsfile(self, pattern='*'):
        """List files inside this directory sorted by name.
        An optional glob pattern could be provided. This pattern is
        concatenated to the path directory.
        Eg: d.lsfile('*/*.pdf') list all files with pdf extension inside
        a subdirectory of directory d.

        Returns:
            List[FilePath]
        """
        pattern = os.path.join(self.path, pattern)
        files = [FilePath(f) for f in glob.glob(pattern) if os.path.isfile(f)]
        return sorted(files)

    def lsdir(self):
        """List directories inside this directory sorted by name.

        Returns:
            List[Directory]
        """
        pattern = os.path.join(self.path, '*')
        dirs = [Directory(f) for f in glob.glob(pattern) if os.path.isdir(f)]
        return sorted(dirs)

    def find_file(self, filename):
        """Finds a file by name in this directory.

        Returns:
            Optional[FilePath]: Path to file if there exists some file
                with the provided name or None otherwise.
        """
        files = self.lsfile(filename)
        return files[0] if len(files) > 0 else None
