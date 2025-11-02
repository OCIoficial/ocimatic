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


@dataclass
class Testplan:
    version: Version
    path: Path | None
    subtasks: list[tuple[testplan.SubtaskHeader, list[testplan.Item]]]
    diagnostics: list[types.Diagnostic]

    def path_at_position(self, pos: types.Position) -> testplan.Path | None:
        for _, items in self.subtasks:
            for item in items:
                match item:
                    case testplan.Script(cmd=cmd):
                        t = cmd
                    case testplan.Validator(path=path):
                        t = path
                    case _:
                        continue
                range = map_range(t.range)
                if range.start <= pos <= range.end:
                    return self.script_path(t.lexeme)

    def script_path(self, path: str) -> Path | None:
        if self.path is None:
            return

        return self.path.parent / path


class OcimaticServer(LanguageServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        self.testplans: dict[URI, Testplan] = {}

    def parse(self, version: Version, doc: TextDocument) -> None:
        parser = testplan.Parser()
        parser.parse(doc.source)

        path = to_fs_path(doc.uri)

        self.testplans[doc.uri] = Testplan(
            version=version,
            path=Path(path) if path else None,
            subtasks=parser.subtasks,
            diagnostics=[
                types.Diagnostic(
                    range=map_range(error.range),
                    message=error.msg,
                    severity=types.DiagnosticSeverity.Error,
                )
                for error in parser.errors
                if error.range is not None
            ],
        )


def map_position(pos: testplan.Position) -> types.Position:
    return types.Position(line=pos.line, character=pos.column)


def map_range(range: testplan.Range) -> types.Range:
    return types.Range(start=map_position(range.start), end=map_position(range.end))


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

    if (target_uri := from_fs_path(str(path))) is None:
        return

    return types.Location(
        uri=target_uri,
        range=types.Range(
            start=types.Position(line=0, character=0),
            end=types.Position(line=0, character=0),
        ),
    )
