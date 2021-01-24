import glob
import os
import shutil
from functools import total_ordering
from tempfile import mkdtemp, mkstemp
from typing import List, Optional, Tuple, Union

from ocimatic import ui


def change_directory():
    """Changes directory to the contest root and returns the absolute path of the
    last directory before reaching the root, this correspond to the directory
    of the problem in which ocimatic was called. If the function reach system
    root the program exists with an error.

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


class FilePath:
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

    def isdir(self) -> bool:
        return os.path.isdir(str(self))

    def open(self, mode: str):
        return open(str(self), mode)

    def copy(self, dest: Union['Directory', 'FilePath']) -> None:
        """Copy file in this path.
        Args:
           dest (FilePath|Directory): destination path
        """
        shutil.copy2(str(self), str(dest))

    def create_dir(self) -> 'Directory':
        os.mkdir(str(self))
        return Directory(str(self))

    def get_or_create_dir(self):
        if not self.exists():
            self.create_dir()
        return Directory(str(self))

    def remove(self) -> None:
        """Removes file in this path from the filesystem"""
        os.remove(str(self))

    @property
    def name(self) -> str:
        """Name of the file.
        Returns:
           str
        """
        return self._filename

    @property
    def ext(self) -> str:
        """str: File extension beginning with dot. """
        return self.splitext()[1]

    def rootname(self) -> str:
        """Return filename without extension"""
        return self.splitext()[0]

    def __str__(self) -> str:
        return os.path.join(self._directory, self._filename)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FilePath):
            return False
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, FilePath):
            return False
        return str(self) < str(other)

    def chext(self, ext: str) -> 'FilePath':
        """Change extension of file

        Args:
            ext (string): new extension including dot.

        Returns:
            FilePath: file path with the new extension
        """
        return FilePath(os.path.join(self._directory, self.rootname() + ext))

    def splitext(self) -> Tuple[str, str]:
        """Split filename in root name and extension.

        Returns:
            (str, str): A pair where the first component corresponds
                to root name and the second to the extension.
        """
        return os.path.splitext(self._filename)

    def mtime(self) -> float:
        """Returns modification time. If the file does not exists in
        the file system this functions returns the oldest possible date.

        Returns:
            float: Numbers of seconds since the epoch or -inf if the file
                does not exists.
        """
        if self.exists():
            return os.path.getmtime(str(self))
        return float('-Inf')

    def directory(self) -> 'Directory':
        """Returns the directory where the file resides.

        Returns:
            Directory
        """
        return Directory(self._directory)

    def exists(self) -> bool:
        """Returns true if the file actually exists in the file system

        Returns:
            bool
        """
        return os.path.exists(str(self))


@total_ordering
class Directory:
    """Represent a directory in the filesystem. The directory must always
    exist.
    """
    @staticmethod
    def tmpdir() -> 'Directory':
        """Returns a temporary directory. The user is responsible
        of deleting the directory after using it.
        Returns:
            Directory
        """
        return Directory(mkdtemp())

    @staticmethod
    def getcwd() -> 'Directory':
        return Directory(os.getcwd())

    def __init__(self, path: str, create: bool = False):
        """Produces an assertion error if the directory does not exist (unless create is true)
        or if the path does not correspond to a directory.
        Args:
            path (str): Full path to directory.
            create (bool): Whether the directory should be created if it doesn't exists.
        """
        if os.path.exists(path):
            assert os.path.isdir(path)
        elif create:
            os.makedirs(path)
        else:
            assert os.path.exists(path)

        self._path = os.path.abspath(path)

    def mkdir(self, name) -> 'Directory':
        """Create new directory name inside this directory
        Args:
            parent (Directory): Parent directory
            name (str): name of the new directory
        Returns:
            Directory: the created directory
        """
        path = os.path.join(str(self.path()), name)
        assert not os.path.exists(path)
        os.mkdir(path)
        return Directory(path)

    def clear(self) -> None:
        """Remove all files in this directory recursively but keeps the directory
        """
        for f in self.lsfile():
            f.remove()
        for subdir in self.lsdir():
            subdir.rmtree()

    def __str__(self) -> str:
        return str(self.path())

    def rmtree(self) -> None:
        """Removes this directory from the filesystem."""
        shutil.rmtree(str(self.path()))

    @property
    def basename(self) -> str:
        """str: Name of this directory"""
        return os.path.basename(self._path)

    def path(self) -> FilePath:
        """FilePath: Full path to directory."""
        return FilePath(self._path)

    def chdir(self, *subdirs: str, create: bool = False) -> 'Directory':
        """Changes directory. If the new directory does not exists
        this functions produces an assertion error, unless create is true.

        Args:
            paths (List[str]):

        Returns:
            Directory: The new directory.
        """
        return Directory(os.path.join(str(self.path()), *subdirs), create=create)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Directory):
            return False
        return self.path() == other.path()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Directory):
            return False
        return self.path() < other.path()

    def lsfile(self, pattern: str = '*') -> List[FilePath]:
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

    def lsdir(self, pattern: str = '*') -> List['Directory']:
        """List directories inside this directory sorted by name.

        Returns:
            List[Directory]
        """
        pattern = os.path.join(str(self.path()), pattern)
        dirs = [Directory(f) for f in glob.glob(pattern) if os.path.isdir(f)]
        return sorted(dirs)

    def ls(self, pattern: str = '*') -> List[FilePath]:
        pattern = os.path.join(str(self.path()), pattern)
        paths = [FilePath(f) for f in glob.glob(pattern)]
        return sorted(paths)

    def find(self, name: str) -> Optional[FilePath]:
        files = self.ls(name)
        return files[0] if files else None

    def find_file(self, filename: str) -> Optional[FilePath]:
        """Finds a file by name in this directory.

        Returns:
            Optional[FilePath]: Path to file if there exists some file
                with the provided name or None otherwise.
        """
        files = self.lsfile(filename)
        return files[0] if files else None

    def copy_tree(self, dst: FilePath, ignore: Optional[List[str]] = None) -> None:
        """
        Copy recursively all content from directory to destination and create
        dest if it does not exists.
        Args:
            dest (FilePath): destination folder
        """
        ignore_pats = shutil.ignore_patterns(*ignore) if ignore is not None else None
        shutil.copytree(str(self.path()), str(dst), symlinks=True, ignore=ignore_pats)
