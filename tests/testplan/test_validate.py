from __future__ import annotations

from ocimatic.testplan import validate

from .harness import assert_parses_ok, assert_validation_errors


def test_valid_testplan() -> None:
    subtasks = assert_parses_ok(r"""
        [Subtask 1]
          @validator validator.py
          small ; echo 1 2
        [Subtask 2]
          @extends subtask 1
        [Subtask 3]
          @extends subtask 1
          @extends subtask 2
    """)
    assert validate(subtasks) == []


def test_subtask_zero() -> None:
    assert_validation_errors(r"""
          [Subtask 0]
        #~^^^^^^^^^^^ found [Subtask 0], but [Subtask 1] was expected
          small ; echo 1 2
    """)


def test_subtasks_out_of_order() -> None:
    assert_validation_errors(r"""
        [Subtask 1]
          [Subtask 3]
        #~^^^^^^^^^^^ found [Subtask 3], but [Subtask 2] was expected
          [Subtask 2]
        #~^^^^^^^^^^^ found [Subtask 2], but [Subtask 3] was expected
    """)


def test_multiple_validators() -> None:
    assert_validation_errors(r"""
        [Subtask 1]
          @validator a.py
          @validator b.py
        #~^^^^^^^^^^^^^^^ multiple @validator directives found for the same subtask
    """)


def test_invalid_extends() -> None:
    assert_validation_errors(r"""
        [Subtask 1]
        [Subtask 2]
          @extends subtask 1
          @extends subtask 1
        #~^^^^^^^^^^^^^^^^^^ cannot extends twice from the same subtask: `@extends subtask 1`
          @extends subtask 2
        #~^^^^^^^^^^^^^^^^^^ a subtask cannot extend itself: `@extends subtask 2`
          @extends subtask 3
        #~^^^^^^^^^^^^^^^^^^ invalid subtask 3: `@extends subtask 3`
    """)


def test_extends_cycle() -> None:
    assert_validation_errors(r"""
        [Subtask 1]
          @extends subtask 2
        [Subtask 2]
          @extends subtask 1
        #~^^^^^^^^^^^^^^^^^^ `@extends subtask 1` creates a cycle in the extends graph
    """)
