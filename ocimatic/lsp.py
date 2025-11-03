from dataclasses import dataclass
import fnmatch
import logging
from pathlib import Path
from typing import Any
import uuid
from lsprotocol import types

from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument
from pygls.uris import to_fs_path, from_fs_path

from ocimatic.testplan import (
    Parser,
    SubtaskHeader,
    Item,
    Range,
    Position,
    Script,
    Validator,
)

type URI = str
type Version = int

logging.basicConfig(level=logging.INFO, format="%(message)s")

FILE_NOT_FOUND = 0
SYNTAX_ERROR = 1


@dataclass
class Testplan:
    version: Version
    paths: dict[Path, list[types.Range]]
    """All paths (i.e. files) in a testplan and the list of ranges they appear in."""

    subtasks: list[tuple[SubtaskHeader, list[Item]]]
    syntax_errors: list[types.Diagnostic]

    def path_at_position(self, pos: types.Position) -> Path | None:
        for path, ranges in self.paths.items():
            for range in ranges:
                if range.start <= pos <= range.end:
                    return path

    def all_diagnostics(self) -> list[types.Diagnostic]:
        return [*self.syntax_errors, *self.file_not_founds()]

    def file_not_founds(self) -> list[types.Diagnostic]:
        return [
            types.Diagnostic(
                code=FILE_NOT_FOUND,
                range=range,
                message="file not found",
                severity=types.DiagnosticSeverity.Error,
                data=path,  # we recover the data in the quick fix
            )
            for path, ranges in self.paths.items()
            if not path.exists()
            for range in ranges
        ]


class OcimaticServer(LanguageServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        self.testplans: dict[URI, Testplan] = {}

    def parse(self, version: Version, doc: TextDocument) -> None:
        parser = Parser()
        parser.parse(doc.source)

        if testplan_path := to_fs_path(doc.uri):
            paths = _find_paths(Path(testplan_path).parent, parser.subtasks)
        else:
            paths = {}

        self.testplans[doc.uri] = Testplan(
            version=version,
            paths=paths,
            subtasks=parser.subtasks,
            syntax_errors=[
                types.Diagnostic(
                    code=SYNTAX_ERROR,
                    range=_map_range(error.range),
                    message=error.msg,
                    severity=types.DiagnosticSeverity.Error,
                )
                for error in parser.errors
                if error.range is not None
            ],
        )


def _find_paths(
    parent: Path,
    subtasks: list[tuple[SubtaskHeader, list[Item]]],
) -> dict[Path, list[types.Range]]:
    paths: dict[Path, list[types.Range]] = {}
    for _, items in subtasks:
        for item in items:
            match item:
                case Script(cmd=cmd):
                    t = cmd
                case Validator(path=path):
                    t = path
                case _:
                    continue
            path = Path(parent, t.lexeme)
            paths.setdefault(path, []).append(_map_range(t.range))
    return paths


def _map_position(pos: Position) -> types.Position:
    return types.Position(line=pos.line, character=pos.column)


def _map_range(range: Range) -> types.Range:
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


@server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
def did_close(ls: OcimaticServer, params: types.DidCloseTextDocumentParams) -> None:
    ls.testplans.pop(params.text_document.uri)


WATCHERS = [
    types.FileSystemWatcher(
        glob_pattern="**/*.py",
        kind=types.WatchKind.Create | types.WatchKind.Delete,
    ),
    types.FileSystemWatcher(
        glob_pattern="**/*.cpp",
        kind=types.WatchKind.Create | types.WatchKind.Delete,
    ),
]


@server.feature(types.INITIALIZED)
def initialized(ls: OcimaticServer, params: types.InitializeParams) -> None:
    ls.client_register_capability(
        types.RegistrationParams(
            registrations=[
                types.Registration(
                    id=str(uuid.uuid4()),
                    method=types.WORKSPACE_DID_CHANGE_WATCHED_FILES,
                    register_options=types.DidChangeWatchedFilesRegistrationOptions(
                        watchers=WATCHERS,
                    ),
                ),
            ],
        ),
    )


def _matches_watch_kind(change_type: types.FileChangeType, watch_kind: int) -> bool:
    bit_value = 1 << (change_type.value - 1)  # 1→1, 2→2, 3→4
    return (watch_kind & bit_value) != 0


def _matches_watcher(ev: types.FileEvent, watcher: types.FileSystemWatcher) -> bool:
    if not (kind := watcher.kind) or not _matches_watch_kind(ev.type, kind):
        return False
    if not (path := to_fs_path(ev.uri)):
        return False
    if not isinstance(pat := watcher.glob_pattern, str):
        return False
    if not fnmatch.fnmatch(path, pat):
        return False
    return False


def _is_watched(ev: types.FileEvent) -> bool:
    return any(_matches_watcher(ev, watcher) for watcher in WATCHERS)


@server.feature(types.WORKSPACE_DID_CHANGE_WATCHED_FILES)
async def did_change_watched_files(
    ls: OcimaticServer,
    params: types.DidChangeWatchedFilesParams,
) -> None:
    # zed doesn't honor the watchers so we check here
    if any(not _is_watched(f) for f in params.changes):
        return
    await ls.workspace_diagnostic_refresh_async(None)


@server.feature(
    types.TEXT_DOCUMENT_DIAGNOSTIC,
    types.DiagnosticRegistrationOptions(
        identifier="pull-diagnostics",
        inter_file_dependencies=False,
        workspace_diagnostics=False,
    ),
)
def document_diagnostic(
    ls: OcimaticServer,
    params: types.DocumentDiagnosticParams,
) -> types.DocumentDiagnosticReport | None:
    if (uri := params.text_document.uri) not in ls.testplans:
        return

    testplan = ls.testplans[uri]

    return types.RelatedFullDocumentDiagnosticReport(items=testplan.all_diagnostics())


@server.feature(types.WORKSPACE_DIAGNOSTIC)
def workspace_diagnostic(
    ls: OcimaticServer,
    _params: types.WorkspaceDiagnosticParams,
) -> types.WorkspaceDiagnosticReport | None:
    items: list[types.WorkspaceDocumentDiagnosticReport] = []
    for uri, testplan in ls.testplans.items():
        items.append(
            types.WorkspaceFullDocumentDiagnosticReport(
                uri=uri,
                version=testplan.version,
                items=testplan.all_diagnostics(),
            ),
        )

    return types.WorkspaceDiagnosticReport(items=items)


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
def code_actions(params: types.CodeActionParams) -> list[types.CodeAction] | None:
    for diagnostic in params.context.diagnostics:
        if (
            diagnostic.code == FILE_NOT_FOUND
            and isinstance(diagnostic.data, str)
            and (uri := from_fs_path(diagnostic.data))
        ):
            return [
                types.CodeAction(
                    title="Create File",
                    kind=types.CodeActionKind.QuickFix,
                    edit=types.WorkspaceEdit(
                        document_changes=[types.CreateFile(uri=uri)],
                    ),
                    diagnostics=[diagnostic],
                ),
            ]
