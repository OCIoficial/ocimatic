import glob
import os
import shutil
from functools import total_ordering
from tempfile import mkstemp, mkdtemp
from distutils.dir_util import copy_tree

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


class FilePath(object):
    """Represents a path to a file. The file may not exist in the
    file system.
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
            if isinstance(arg1, Directory):
                self._directory = str(arg1.path())
            else:
                self._directory = arg1
            self._filename = arg2
        else:
            dirname = os.path.dirname(arg1)
            self._directory = dirname
            self._filename = os.path.basename(arg1)

    def isdir(self):
        return os.path.isdir(str(self))

    def open(self, mode):
        return open(str(self), mode)

    def copy(self, dest):
        """Copy file in this path.
        Args:
           dest (FilePath|Directory): destination path
        """
        shutil.copy2(str(self), str(dest))

    def create_dir(self):
        os.mkdir(str(self))
        return Directory(str(self))

    def get_or_create_dir(self):
        if not self.exists():
            self.create_dir()
        return Directory(str(self))

    def remove(self):
        """Removes file in this path from the filesystem"""
        os.remove(str(self))

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

    def rootname(self):
        """Return filename without extension.
        Args:
            full (bool): if true returns fullpath."""
        return self.splitext()[0]

    def __str__(self):
        return os.path.join(self._directory, self._filename)

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def chext(self, ext):
        """Change extension of file

        Args:
            ext (string): new extension including dot.

        Returns:
            FilePath: file path with the new extension
        """
        return FilePath(os.path.join(self._directory, self.rootname()+ext))

    def splitext(self):
        """Split filename in root name and extension.

        Returns:
            (str, str): A pair where the first component corresponds
                to root name and the second to the extension.
        """
        return os.path.splitext(self._filename)

    def mtime(self):
        """Returns modification time. If the file does not exists in
        the file system this functions returns the oldest possible date.

        Returns:
            float: Numbers of seconds since the epoch or -inf if the file
                does not exists.
        """
        if self.exists():
            return os.path.getmtime(str(self))
        else:
            return float('-Inf')

    def directory(self):
        """Returns the directory where the file resides.

        Returns:
            Directory
        """
        return Directory(self._directory)

    def exists(self):
        """Returns true if the file actually exists in the file system

        Returns:
            bool
        """
        return os.path.exists(str(self))


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

    @staticmethod
    def getcwd():
        return Directory(os.getcwd())

    def __init__(self, path):
        """Produces an assertion error if the directory does not exist
        or if the path does not correspond to a directory.
        Args:
            path (str): Full path to directory.
        """
        assert(os.path.exists(path))
        assert(os.path.isdir(path))
        self._path = os.path.abspath(path)

    def mkdir(self, name):
        """Create new directory name inside this directory
        Args:
            parent (Directory): Parent directory
            name (str): name of the new directory
        Returns:
            Directory: the created directory
        """
        path = os.path.join(str(self.path()), name)
        assert(not os.path.exists(path))
        os.mkdir(path)
        return Directory(path)

    def clear(self):
        """Remove all files in this directory recursively but keeps the directory
        """
        for f in self.lsfile():
            f.remove()
        for subdir in self.lsdir():
            subdir.rmtree()

    def __str__(self):
        return str(self.path())

    def rmtree(self):
        """Removes this directory from the filesystem."""
        shutil.rmtree(str(self.path()))

    @property
    def basename(self):
        """str: Name of this directory"""
        return os.path.basename(self._path)

    def path(self):
        """FilePath: Full path to directory."""
        return FilePath(self._path)

    def chdir(self, *subdirs):
        """Changes directory. If the new directory does not exists
        this functions produces an assertion error.

        Args:
            paths (List[str]):

        Returns:
            Directory: The new directory.
        """
        return Directory(os.path.join(str(self.path()), *subdirs))

    def __eq__(self, other):
        return self.path() == other.path()

    def __lt__(self, other):
        return self.path() < other.path()

    def lsfile(self, pattern='*'):
        """List files inside this directory sorted by name.
        An optional glob pattern could be provided. This pattern is
        concatenated to the path directory.
        Eg: d.lsfile('*/*.pdf') list all files with pdf extension inside
        a subdirectory of directory d.

        Returns:
            List[FilePath]
        """
        pattern = os.path.join(str(self.path()), pattern)
        files = [FilePath(f) for f in glob.glob(pattern) if os.path.isfile(f)]
        return sorted(files)

    def lsdir(self, pattern='*'):
        """List directories inside this directory sorted by name.

        Returns:
            List[Directory]
        """
        pattern = os.path.join(str(self.path()), pattern)
        dirs = [Directory(f) for f in glob.glob(pattern) if os.path.isdir(f)]
        return sorted(dirs)

    def ls(self, pattern='*'):
        pattern = os.path.join(str(self.path()), pattern)
        paths = [FilePath(f) for f in glob.glob(pattern)]
        return sorted(paths)

    def find(self, name):
        files = self.ls(name)
        return files[0] if len(files) > 0 else None

    def find_file(self, filename):
        """Finds a file by name in this directory.

        Returns:
            Optional[FilePath]: Path to file if there exists some file
                with the provided name or None otherwise.
        """
        files = self.lsfile(filename)
        return files[0] if len(files) > 0 else None

    def copy_tree(self, dest):
        """
        Copy recursively all content from directory to destination and create
        dest if it does not exists.
        Args:
            dest (FilePath): destination folder
        """
        copy_tree(str(self.path()), str(dest), preserve_symlinks=1)
