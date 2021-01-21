import random
import re
import shlex
import shutil
import string
import subprocess

import ocimatic
from ocimatic import ui
from ocimatic.compilers import CppCompiler, JavaCompiler
from ocimatic.filesystem import Directory, FilePath
from ocimatic.runnable import Runnable


class Dataset:
    """Test data"""
    def __init__(self, directory, sampledata=None, in_ext='.in', sol_ext='.sol'):
        """
        Args:
            directory (Directory): dataset directory.
            sampledata (Optional[SampleData]): optional sampledata
        """
        self._directory = directory
        self._in_ext = in_ext
        self._sol_ext = sol_ext
        self._subtasks = [Subtask(d, in_ext, sol_ext) for d in directory.lsdir()]
        self._sampledata = sampledata

    def gen_expected(self, runnable, sample=False):
        for subtask in self._subtasks:
            subtask.gen_expected(runnable)
        if sample:
            self._sampledata.gen_expected(runnable)

    def run(self, runnable, checker, check=False, sample=False):
        for subtask in self._subtasks:
            subtask.run(runnable, checker, check=check)
        if sample:
            self._sampledata.run(runnable, checker, check=check)

    def mtime(self):
        mtime = -1
        for subtask in self._subtasks:
            mtime = max(mtime, subtask.mtime())
        return mtime

    def compress(self, in_ext=None, sol_ext=None, random_sort=False):
        """Compress all test cases in this dataset in a single zip file.
        The basename of the corresponding subtask subdirectory is prepended
        to each file.
        """
        in_ext = in_ext or self._in_ext
        sol_ext = sol_ext or self._sol_ext
        dst_file = FilePath(self._directory, 'data.zip')
        if dst_file.exists() and dst_file.mtime() >= self.mtime():
            return True

        tmpdir = Directory.tmpdir()
        try:
            copied = 0
            for subtask in self._subtasks:
                copied += subtask.copy_to(tmpdir, random_sort=random_sort)

            if not copied:
                # ui.show_message("Warning", "no files in dataset", ui.WARNING)
                return True

            cmd = 'cd %s && zip data.zip *%s *%s' % (tmpdir, in_ext, sol_ext)
            st = subprocess.call(cmd, stdout=subprocess.DEVNULL, shell=True)
            FilePath(tmpdir, 'data.zip').copy(dst_file)
        finally:
            tmpdir.rmtree()

        return st == 0

    def count(self):
        return [st.count() for st in self._subtasks]

    def normalize(self):
        for subtask in self._subtasks:
            subtask.normalize()
        self._sampledata.normalize()


class Subtask:
    def __init__(self, directory, in_ext='.in', sol_ext='.sol'):
        self._tests = []
        self._in_ext = in_ext
        self._sol_ext = sol_ext
        for f in directory.lsfile('*' + self._in_ext):
            self._tests.append(Test(f, f.chext(sol_ext)))
        self._name = directory.basename

    def copy_to(self, directory, random_sort=False):
        copied = 0
        for test in self._tests:
            if test.expected_path.exists():
                # Sort testcases withing a subtask randomly
                if random_sort:
                    choices = string.ascii_lowercase
                    rnd_str = ''.join(random.choice(choices) for _ in range(3))
                    in_name = "%s-%s-%s" % (self._name, rnd_str, test.in_path.name)
                    sol_name = "%s-%s-%s" % (self._name, rnd_str, test.expected_path.name)
                else:
                    in_name = "%s-%s" % (self._name, test.in_path.name)
                    sol_name = "%s-%s" % (self._name, test.expected_path.name)
                test.in_path.copy(FilePath(directory, in_name))
                test.expected_path.copy(FilePath(directory, sol_name))
                copied += 1
        return copied

    def mtime(self):
        mtime = -1
        for test in self._tests:
            mtime = max(mtime, test.mtime())
        return mtime

    def normalize(self):
        for test in self._tests:
            test.normalize()

    def count(self):
        return len(self._tests)

    @ui.workgroup()
    def run(self, runnable, checker, check=False):
        for test in self._tests:
            test.run(runnable, checker, check=check)

    @ui.workgroup()
    def gen_expected(self, runnable):
        for test in self._tests:
            test.gen_expected(runnable)

    def __str__(self):
        return self._name


class SampleData(Subtask):
    # FIXME: this shouldn't inherit directly from Subtask as the initializer is completely different.
    # Maybe both should a have a common parent.
    def __init__(self, statement, in_ext='.in', sol_ext='.sol'):  # pylint: disable=super-init-not-called
        self._in_ext = in_ext
        self._sol_ext = sol_ext
        tests = statement.io_samples() if statement else []
        self._tests = [Test(f.chext(in_ext), f.chext(sol_ext)) for f in tests]

    def __str__(self):
        return 'Sample'


class Test:
    """A single test file. Expected output file may not exist"""
    def __init__(self, in_path, expected_path):
        """
        Args:
            in_path (FilePath)
            expected_path (FilePath)
        """
        assert in_path.exists()
        self._in_path = in_path
        self._expected_path = expected_path

    def __str__(self):
        return str(self._in_path)

    def mtime(self):
        if self._expected_path.exists():
            return max(self._in_path.mtime(), self._expected_path.mtime())
        return self._in_path.mtime()

    @property
    def directory(self):
        """Directory: directory where this test reside"""
        return self._in_path.directory()

    @ui.work('Gen')
    def gen_expected(self, runnable):
        """Run binary with this test as input to generate expected output file
        Args:
            runnable (Runnable)
        Returns:
            (bool, msg): A tuple containing status and result message.
        """
        (st, _, errmsg) = runnable.run(self.in_path,
                                       self.expected_path,
                                       timeout=ocimatic.config['timeout'])
        msg = 'OK' if st else errmsg
        return (st, msg)

    @ui.work('Run')
    def run(self, runnable, checker, check=False):
        """Run runnable whit this test as input and check output correctness
        Args:
            runnable (Runnable)
            checker (Checker): Checker to check outcome
            check  (bool): If true this only report if expected output
                correspond to binary execution output.
        """
        out_path = FilePath.tmpfile()
        if not self.expected_path.exists():
            out_path.remove()
            return (False, 'No expected output file')

        (st, time, errmsg) = runnable.run(self.in_path,
                                          out_path,
                                          timeout=ocimatic.config['timeout'])

        # Execution failed
        if not st:
            if check:
                return (st, errmsg)
            return (st, '%s' % errmsg)

        (st, outcome, checkmsg) = checker(self.in_path, self.expected_path, out_path)
        # Checker failed
        if not st:
            msg = 'Failed to run checker: %s' % checkmsg
            return (st, msg)

        st = outcome == 1.0
        if check:
            msg = 'OK' if st else 'FAILED'
            return (st, msg)

        msg = '%s [%.2fs]' % (outcome, time)
        if checkmsg:
            msg += ' - %s' % checkmsg
        return (st, msg)

    @property
    def in_path(self):
        """FilePath: Input file path"""
        return self._in_path

    @property
    def expected_path(self):
        """FilePath: Expected output file path."""
        return self._expected_path

    @ui.work('Normalize')
    def normalize(self):
        if not shutil.which('dos2unix'):
            return (False, 'Cannot find dos2unix')
        if not shutil.which('sed'):
            return (False, 'Cannot find sed')
        tounix_input = 'dos2unix "%s"' % self.in_path
        tounix_expected = 'dos2unix "%s"' % self.expected_path
        sed_input = "sed -i -e '$a\\' \"%s\"" % self.in_path
        sed_expected = "sed -i -e '$a\\' \"%s\"" % self.expected_path
        null = subprocess.DEVNULL
        st = subprocess.call(tounix_input, stdout=null, stderr=null, shell=True)
        st += subprocess.call(sed_input, stdout=null, stderr=null, shell=True)
        if self.expected_path.exists():
            st += subprocess.call(tounix_expected, stdout=null, stderr=null, shell=True)
            st += subprocess.call(sed_expected, stdout=null, stderr=null, shell=True)
        return (st == 0, 'OK' if st == 0 else 'FAILED')


# FIXME: Refactor class. This should allow to re-enable some pylint checks
class DatasetPlan:
    """Functionality to read and run a plan for generating dataset."""
    def __init__(self, directory, task_directory, dataset_directory, filename='testplan.txt'):
        self._directory = directory
        self._testplan_path = FilePath(directory, filename)
        if not self._testplan_path.exists():
            ui.fatal_error('No such file plan for creating dataset: "%s"' % self._testplan_path)
        self._task_directory = task_directory
        self._dataset_directory = dataset_directory
        self._cpp_compiler = CppCompiler()
        self._java_compiler = JavaCompiler()

    def test_filepath(self, stn, group, i):
        st_dir = FilePath(self._dataset_directory, 'st%d' % stn).get_or_create_dir()
        return FilePath(st_dir, '%s-%d.in' % (group, i))

    def validate_input(self):
        (_, cmds) = self.parse_file()
        for (st, subtask) in sorted(cmds.items()):
            self.validate_subtask(st, subtask)

    @ui.workgroup('Subtask {1}')
    def validate_subtask(self, stn, subtask):
        validator = None
        if subtask['validator']:
            (validator, msg) = self.build_validator(subtask['validator'])
            if validator is None:
                ui.show_message('Warning', 'Failed to build validator: %s' % msg, ui.WARNING)
        else:
            ui.show_message('Info', 'No validator specified', ui.INFO)
        if validator:
            for (group, tests) in sorted(subtask['groups'].items()):
                for (i, _) in enumerate(tests, 1):
                    test_file = self.test_filepath(stn, group, i)
                    self.validate_test_input(test_file, validator)

    @ui.work('Validating', '{1}')
    def validate_test_input(self, test_file, validator):
        if not test_file.exists():
            return False, 'Test file does not exist'
        (st, _time, msg) = validator.run(test_file, None)
        return st, msg

    def build_validator(self, source):
        fp = FilePath(self._directory, source)
        if not fp.exists():
            return (None, 'File does not exists.')
        if fp.ext == '.cpp':
            binary = fp.chext('.bin')
            if binary.mtime() < fp.mtime() and not self._cpp_compiler(fp, binary):
                return (None, 'Failed to build validator.')
            return (Runnable(binary), 'OK')
        if fp.ext in ['.py', '.py3']:
            return (Runnable('python3', [str(source)]), 'OK')
        if fp.ext == '.py2':
            return (Runnable('python2', [str(source)]), 'OK')
        return (None, 'Not supported source file.')

    def run(self):
        (subtasks, cmds) = self.parse_file()

        for stn in range(1, subtasks + 1):
            dire = FilePath(self._dataset_directory, 'st%d' % stn).get_or_create_dir()
            dire.clear()

        if not cmds:
            ui.show_message("Warning", 'no commands were executed for the plan.', ui.WARNING)

        for (stn, subtask) in sorted(cmds.items()):
            self.run_subtask(stn, subtask)

    @ui.workgroup('Subtask {1}')
    def run_subtask(self, stn, subtask):
        groups = subtask['groups']
        for (group, tests) in sorted(groups.items()):
            for (i, test) in enumerate(tests, 1):
                cmd = test['cmd']
                test_file = self.test_filepath(stn, group, i)
                if cmd == 'copy':
                    self.copy(test['file'], test_file)
                elif cmd == 'echo':
                    self.echo(test['args'], test_file)
                else:
                    test['args'].insert(0, '%s-%s-%s' % (stn, group, i))
                    source = FilePath(self._directory, test['source'])
                    if cmd == 'cpp':
                        self.run_cpp_generator(source, test['args'], test_file)
                    elif cmd in ['py', 'py2', 'py3']:
                        self.run_py_generator(source, test['args'], test_file, cmd)
                    elif cmd == 'java':
                        self.run_java_generator(source, test['args'], test_file)
                    elif cmd == 'run':
                        bin_path = FilePath(self._directory, test['bin'])
                        self.run_bin_generator(bin_path, test['args'], test_file)
                    else:
                        ui.fatal_error('unexpected command when running plan: %s ' % cmd)

    @ui.work('Copy', '{1}')
    def copy(self, src, dst):
        fp = FilePath(self._task_directory, src)
        if not fp.exists():
            return (False, 'No such file')
        try:
            fp.copy(dst)
            (st, msg) = (True, 'OK')
            return (st, msg)
        except Exception:  # pylint: disable=broad-except
            return (False, 'Error when copying file')

    @ui.work('Echo', '{1}')
    def echo(self, args, dst):
        with dst.open('w') as test_file:
            test_file.write(' '.join(args) + '\n')
            (st, msg) = (True, 'Ok')
            return (st, msg)

    @ui.work('Gen', '{1}')
    def run_cpp_generator(self, source, args, dst):
        if not source.exists():
            return (False, 'No such file')
        binary = source.chext('.bin')
        if binary.mtime() < source.mtime():
            st = self._cpp_compiler(source, binary)
            if not st:
                return (st, 'Failed to build generator')

        (st, _time, msg) = Runnable(binary).run(None, dst, args)
        return (st, msg)

    @ui.work('Gen', '{1}')
    def run_py_generator(self, source, args, dst, cmd):
        if not source.exists():
            return (False, 'No such file')
        python = 'python2' if cmd == 'py2' else 'python3'
        (st, _time, msg) = Runnable(python, [str(source)]).run(None, dst, args)
        # print(' '.join(args))
        return (st, msg)

    @ui.work('Gen', '{1}')
    def run_java_generator(self, source, args, dst):
        if not source.exists():
            return (False, 'No such file')
        bytecode = source.chext('.class')
        if bytecode.mtime() < source.mtime():
            st = self._java_compiler(source)
            if not st:
                return (st, 'Failed to build generator')

        classname = bytecode.rootname()
        classpath = str(bytecode.directory().path())
        (st, _time, msg) = Runnable('java', ['-cp', classpath, classname]).run(None, dst, args)
        return (st, msg)

    @ui.work('Gen', '{1}')
    def run_bin_generator(self, bin_path, args, dst):
        if not bin_path.exists():
            return (False, 'No such file')
        if not Runnable.is_callable(bin_path):
            return (False, 'Cannot run file, it may not have correct permissions')
        (st, _time, msg) = Runnable(bin_path).run(None, dst, args)
        return (st, msg)

    def parse_file(self):  # pylint: disable=too-many-locals,too-many-branches
        """
        Args:
            path (FilePath)
        """
        cmds = {}
        st = 0
        for (lineno, line) in enumerate(self._testplan_path.open('r').readlines(), 1):
            line = line.strip()
            subtask_header = re.compile(r'\s*\[\s*Subtask\s*(\d+)\s*(?:-\s*([^\]\s]+))?\s*\]\s*')
            cmd_line = re.compile(r'\s*([^;\s]+)\s*;\s*(\S+)(:?\s+(.*))?')
            comment = re.compile(r'\s*#.*')

            if not line:
                continue
            if not comment.fullmatch(line):
                header_match = subtask_header.fullmatch(line)
                cmd_match = cmd_line.fullmatch(line)
                if header_match:
                    found_st = int(header_match.group(1))
                    validator = header_match.group(2)
                    if st + 1 != found_st:
                        ui.fatal_error('line %d: found subtask %d, but subtask %d was expected' %
                                       (lineno, found_st, st + 1))
                    st += 1
                    cmds[st] = {'validator': validator, 'groups': {}}
                elif cmd_match:
                    if st == 0:
                        ui.fatal_error('line %d: found command before declaring a subtask.' %
                                       lineno)
                    group = cmd_match.group(1)
                    cmd = cmd_match.group(2)
                    args = _parse_args(cmd_match.group(3) or '')
                    if group not in cmds[st]['groups']:
                        cmds[st]['groups'][group] = []

                    if cmd == 'copy':
                        if len(args) > 2:
                            ui.fatal_error('line %d: command copy expects exactly one argument.' %
                                           lineno)
                        cmds[st]['groups'][group].append({
                            'cmd': 'copy',
                            'file': args[0],
                        })
                    elif cmd == 'echo':
                        cmds[st]['groups'][group].append({'cmd': 'echo', 'args': args})
                    else:
                        f = FilePath(self._directory, cmd)
                        if f.ext in ['.cpp', '.java', '.py', '.py2', '.py3']:
                            cmds[st]['groups'][group].append({
                                'cmd': f.ext[1:],
                                'source': cmd,
                                'args': args
                            })
                        else:
                            cmds[st]['groups'][group].append({
                                'cmd': 'run',
                                'bin': cmd,
                                'args': args
                            })
                else:
                    ui.fatal_error('line %d: error while parsing line `%s`\n' % (lineno, line))
        return (st, cmds)


def _parse_args(args):
    args = args.strip()
    return [a.encode().decode("unicode_escape") for a in shlex.split(args)]
