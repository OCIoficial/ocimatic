from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
from lsprotocol import types

from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument
from pygls.uris import to_fs_path, from_fs_path

from ocimatic import testplan

type URI = str
type Version = int

logging.basicConfig(level=logging.INFO, format="%(message)s")

FILE_NOT_FOUND = 0
SYNTAX_ERROR = 1


@dataclass
class Testplan:
    version: Version
    paths: list[tuple[types.Range, Path]]
    subtasks: list[tuple[testplan.SubtaskHeader, list[testplan.Item]]]
    diagnostics: list[types.Diagnostic]

    def path_at_position(self, pos: types.Position) -> Path | None:
        for range, path in self.paths:
            if range.start <= pos <= range.end:
                return path


class OcimaticServer(LanguageServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        self.testplans: dict[URI, Testplan] = {}

    def parse(self, version: Version, doc: TextDocument) -> None:
        parser = testplan.Parser()
        parser.parse(doc.source)

        if testplan_path := to_fs_path(doc.uri):
            paths = _find_paths(Path(testplan_path).parent, parser.subtasks)
        else:
            paths = []

        # Parse errors
        diagnostics = [
            types.Diagnostic(
                code=SYNTAX_ERROR,
                range=_map_range(error.range),
                message=error.msg,
                severity=types.DiagnosticSeverity.Error,
            )
            for error in parser.errors
            if error.range is not None
        ]

        # File not founds
        diagnostics.extend(
            types.Diagnostic(
                code=FILE_NOT_FOUND,
                range=range,
                message="file not found",
                severity=types.DiagnosticSeverity.Error,
                data=path,  # we recover the data in the quick fix
            )
            for range, path in paths
            if not path.exists()
        )

        self.testplans[doc.uri] = Testplan(
            version=version,
            paths=paths,
            subtasks=parser.subtasks,
            diagnostics=diagnostics,
        )


def _find_paths(
    parent: Path,
    subtasks: list[tuple[testplan.SubtaskHeader, list[testplan.Item]]],
) -> list[tuple[types.Range, Path]]:
    paths: list[tuple[types.Range, Path]] = []
    for _, items in subtasks:
        for item in items:
            match item:
                case testplan.Script(cmd=cmd):
                    t = cmd
                case testplan.Validator(path=path):
                    t = path
                case _:
                    continue
            paths.append((_map_range(t.range), Path(parent, t.lexeme)))
    return paths


def _map_position(pos: testplan.Position) -> types.Position:
    return types.Position(line=pos.line, character=pos.column)


def _map_range(range: testplan.Range) -> types.Range:
    return types.Range(start=_map_position(range.start), end=_map_position(range.end))


server = OcimaticServer("ocimatic-language-server", "v1")


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: OcimaticServer, params: types.DidOpenTextDocumentParams) -> None:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse(params.text_document.version, doc)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: OcimaticServer, params: types.DidOpenTextDocumentParams) -> None:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse(params.text_document.version, doc)


@server.feature(
    types.TEXT_DOCUMENT_DIAGNOSTIC,
    types.DiagnosticOptions(
        identifier="pull-diagnostics",
        inter_file_dependencies=False,
        workspace_diagnostics=False,
    ),
)
def document_diagnostic(
    ls: OcimaticServer,
    params: types.DocumentDiagnosticParams,
) -> types.DocumentDiagnosticReport | None:
    # logging.info("%s", params)

    if (uri := params.text_document.uri) not in ls.testplans:
        return

    testplan = ls.testplans[uri]
    result_id = f"{uri}@{testplan.version}"

    if result_id == params.previous_result_id:
        return types.RelatedUnchangedDocumentDiagnosticReport(result_id)

    return types.RelatedFullDocumentDiagnosticReport(
        items=testplan.diagnostics,
        result_id=result_id,
    )


@server.feature(types.TEXT_DOCUMENT_DEFINITION)
def goto_definition(
    ls: OcimaticServer,
    params: types.DefinitionParams,
) -> types.Location | None:
    if (testplan := ls.testplans.get(params.text_document.uri)) is None:
        return

    if (path := testplan.path_at_position(params.position)) is None:
        return

    if not path.exists():
        return

    if (target_uri := from_fs_path(str(path))) is None:
        return

    return types.Location(
        uri=target_uri,
        range=types.Range(
            start=types.Position(line=0, character=0),
            end=types.Position(line=0, character=0),
        ),
    )


@server.feature(
    types.TEXT_DOCUMENT_CODE_ACTION,
    types.CodeActionOptions(code_action_kinds=[types.CodeActionKind.QuickFix]),
)
def code_actions(
    ls: OcimaticServer,
    params: types.CodeActionParams,
) -> types.CodeAction | None:
    for diagnostic in params.context.diagnostics:
        if (
            diagnostic.code == FILE_NOT_FOUND
            and isinstance(diagnostic.data, str)
            and (uri := from_fs_path(diagnostic.data))
        ):
            return types.CodeAction(
                title="create file",
                edit=types.WorkspaceEdit(document_changes=[types.CreateFile(uri=uri)]),
                diagnostics=[diagnostic],
            )
