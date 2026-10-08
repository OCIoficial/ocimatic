from __future__ import annotations

from pathlib import Path

from lsprotocol import types
from pygls.uris import from_fs_path
from pygls.workspace import TextDocument

from ocimatic import lsp


def test_closing_a_document_that_was_never_opened() -> None:
    ls = lsp.OcimaticServer("test", "v1")
    params = types.DidCloseTextDocumentParams(
        text_document=types.TextDocumentIdentifier(uri="file:///never/opened.txt"),
    )
    lsp.did_close(ls, params)
    assert ls.testplans == {}


def test_missing_file_has_create_file_quick_fix(tmp_path: Path) -> None:
    missing = tmp_path / "gen.py"
    range_ = types.Range(
        start=types.Position(line=1, character=8),
        end=types.Position(line=1, character=14),
    )
    # Imported through the module so pytest doesn't try to collect `Testplan` as a test class.
    testplan = lsp.Testplan(
        version=1,
        paths={missing: [range_]},
        subtasks=[],
        errors=[],
    )
    [diagnostic] = testplan.file_not_founds()

    actions = lsp.code_actions(
        types.CodeActionParams(
            text_document=types.TextDocumentIdentifier(uri="file:///testplan.txt"),
            range=range_,
            context=types.CodeActionContext(diagnostics=[diagnostic]),
        ),
    )

    assert actions is not None
    [action] = actions
    assert action.title == "Create File `gen.py`"
    assert action.edit is not None
    uri = from_fs_path(str(missing))
    assert uri is not None
    assert action.edit.document_changes == [types.CreateFile(uri=uri)]


def test_every_missing_file_in_range_has_a_quick_fix(tmp_path: Path) -> None:
    gen = tmp_path / "gen.py"
    validator = tmp_path / "validator.cpp"
    gen_ranges = [_line_range(1), _line_range(4)]
    testplan = lsp.Testplan(
        version=1,
        paths={gen: gen_ranges, validator: [_line_range(2)]},
        subtasks=[],
        errors=[],
    )
    diagnostics = testplan.file_not_founds()

    actions = lsp.code_actions(
        types.CodeActionParams(
            text_document=types.TextDocumentIdentifier(uri="file:///testplan.txt"),
            range=types.Range(start=_line_range(0).start, end=_line_range(5).end),
            context=types.CodeActionContext(diagnostics=diagnostics),
        ),
    )

    # One action per file, fixing every diagnostic for that file.
    assert actions is not None
    assert [(action.title, len(action.diagnostics or [])) for action in actions] == [
        ("Create File `gen.py`", 2),
        ("Create File `validator.cpp`", 1),
    ]


def _line_range(line: int) -> types.Range:
    return types.Range(
        start=types.Position(line=line, character=2),
        end=types.Position(line=line, character=8),
    )


def test_validation_errors_are_reported() -> None:
    ls = lsp.OcimaticServer("test", "v1")
    uri = "file:///task/testplan/testplan.txt"
    ls.parse(1, TextDocument(uri, source="[Subtask 1]\n[Subtask 3]\n"))

    [diagnostic] = ls.testplans[uri].errors
    assert diagnostic.code == lsp.VALIDATION_ERROR
    assert diagnostic.message == "found [Subtask 3], but [Subtask 2] was expected"
    assert diagnostic.range == types.Range(
        start=types.Position(line=1, character=0),
        end=types.Position(line=1, character=11),
    )
