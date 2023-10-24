import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from ocimatic import ui
from ocimatic.checkers import Checker
from ocimatic.dataset import Dataset, DatasetResults, RunMode
from ocimatic.source_code import (BuildError, CppSource, JavaSource, PythonSource, RustSource,
                                  SourceCode)


@dataclass
class SolutionComment:
    should_fail: Set[int]


class Solution:
    """Abstract class to represent a solution
    """

    def __init__(self, source: SourceCode):
        self._source = source

    def check_should_fail(self, results: DatasetResults) -> bool:
        comment = self._source.comment
        return results.check_should_fail(comment.should_fail)

    @staticmethod
    def load_solutions_in_dir(codename: str, dir: Path, managers_dir: Path) -> List['Solution']:
        """Search for solutions in a directory."""
        assert dir.is_dir()
        return [
            solution for file_path in dir.iterdir()
            for solution in [Solution.load(codename, file_path, managers_dir)] if solution
        ]

    @staticmethod
    def load(codename: str, source_path: Path, managers_dir: Path) -> Optional['Solution']:
        if not source_path.is_file():
            return None

        source: SourceCode
        if source_path.suffix == '.cpp':
            grader = Path(managers_dir, 'grader.cpp')
            source = CppSource(source_path,
                               extra_files=[grader] if grader.exists() else [],
                               include=managers_dir)
        elif source_path.suffix == '.java':
            source = JavaSource(codename, source_path)
        elif source_path.suffix == '.py':
            source = PythonSource(source_path)
        elif source_path.suffix == '.rs':
            source = RustSource(source_path)
        else:
            return None

        return Solution(source)

    @ui.solution_group()
    def run(self, dataset: Dataset, checker: Checker,
            mode: RunMode) -> ui.SolutionGroup[Optional[DatasetResults]]:
        """Run this solution for all test cases in the given dataset."""
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            yield ui.Result.fail(short_msg="Failed", long_msg=build_result.msg)
            return None
        else:
            yield ui.Result.success(short_msg="OK")
            return dataset.run(build_result, checker, mode)

    @ui.solution_group()
    def gen_expected(self, dataset: Dataset, sample: bool = False) -> ui.SolutionGroup[None]:
        """Generate expected output files for all test cases in the given dataset
        running this solution."""
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            yield ui.Result.fail(short_msg="Failed", long_msg=build_result.msg)
        else:
            yield ui.Result.success(short_msg="OK")
            dataset.gen_expected(build_result, sample=sample)

    @ui.work('Build')
    def build(self) -> ui.WorkResult:
        """Build solution."""
        result = self._source.build(True)
        if isinstance(result, BuildError):
            return ui.WorkResult.fail(short_msg='Failed', long_msg=result.msg)
        else:
            return ui.WorkResult.success(short_msg='OK')

    @property
    def name(self) -> str:
        return self._source.name

    def __str__(self) -> str:
        return str(self.name)

    @property
    def source(self) -> SourceCode:
        return self._source

    def extract_comment(self) -> SolutionComment | None:
        comment_str = self._source.line_comment_str()
        comment_pattern = rf"\s*{comment_str}\s*@ocimatic\s+should-fail\s*=\s*\[(st\d+\s*(,\s*st\d+)*(\s*,)?\s*)\]"

        first_line = next(open(self._source.file))
        if not first_line:
            return None

        m = re.match(comment_pattern, first_line)
        if not m:
            return None

        should_fail = set(int(st.strip().removeprefix('st')) for st in m.group(1).split(','))
        return SolutionComment(should_fail=should_fail)
