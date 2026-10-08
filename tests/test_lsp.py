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
    assert action.title == "Create File"
    assert action.edit is not None
    uri = from_fs_path(str(missing))
    assert uri is not None
    assert action.edit.document_changes == [types.CreateFile(uri=uri)]


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
