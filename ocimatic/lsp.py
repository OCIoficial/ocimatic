from typing import Any
from lsprotocol import types

from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument

from ocimatic.testplan import Parser
from ocimatic import testplan

type URI = str
type Version = int


class OcimaticServer(LanguageServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        self.diagnostics: dict[URI, tuple[Version, list[types.Diagnostic]]] = {}

    def parse(self, version: Version, doc: TextDocument) -> None:
        parser = Parser()
        parser.parse(doc.source)

        diagnostics = [
            types.Diagnostic(
                range=map_range(error.range),
                message=error.msg,
                severity=types.DiagnosticSeverity.Error,
            )
            for error in parser.errors
            if error.range is not None
        ]

        self.diagnostics[doc.uri] = (version, diagnostics)


def map_position(pos: testplan.Position) -> types.Position:
    return types.Position(line=pos.line, character=pos.column)


def map_range(range: testplan.Range) -> types.Range:
    return types.Range(start=map_position(range.start), end=map_position(range.end))


server = OcimaticServer("ocimatic-lsp", "v1")


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

    if (uri := params.text_document.uri) not in ls.diagnostics:
        return

    version, diagnostics = ls.diagnostics[uri]
    result_id = f"{uri}@{version}"

    if result_id == params.previous_result_id:
        return types.RelatedUnchangedDocumentDiagnosticReport(result_id)

    return types.RelatedFullDocumentDiagnosticReport(
        items=diagnostics,
        result_id=result_id,
    )
