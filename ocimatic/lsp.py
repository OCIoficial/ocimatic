from typing import Any
from lsprotocol import types

from pygls.lsp.server import LanguageServer
import pygls.cli

type URI = str
type Version = int


def start_server() -> None:
    pygls.cli.start_server(server)


class OcimaticServer(LanguageServer):
    """Language server demonstrating "pull-model" diagnostics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        self.diagnostics: dict[URI, tuple[Version, list[types.Diagnostic]]] = {}


server = OcimaticServer("ocimatic-lsp", "v1")


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: OcimaticServer, params: types.DidOpenTextDocumentParams) -> None:
    _doc = ls.workspace.get_text_document(params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: OcimaticServer, params: types.DidOpenTextDocumentParams) -> None:
    _doc = ls.workspace.get_text_document(params.text_document.uri)


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
