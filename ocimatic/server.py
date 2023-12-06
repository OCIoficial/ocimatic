from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

from ansi2html import Ansi2HTMLConverter
from flask import Flask, Response, render_template, request

from ocimatic import core


def ansi2html(ansi: str) -> str:
    return Ansi2HTMLConverter(scheme="mint-terminal", inline=True).convert(
        ansi,
        full=False,
    )


UPLOAD_FOLDER = Path("/tmp", "ocimatic", "server")

contest: core.Contest | None = None
app = Flask(__name__)
app.secret_key = 'M\xf2\xba\xc0\xe3\xe55\xa0"\xff\x96\xba\xb8Jn\xc6#S\xa0t\xda\xb5[\r'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def upload_folder() -> Path:
    UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
    return UPLOAD_FOLDER


@app.route("/", methods=["POST", "GET"])
def server() -> str:
    assert contest
    return render_template("index.html", tasks=contest.tasks)


def save_solution(content: str, suffix: str) -> Path:
    filepath = Path(upload_folder(), f"solution.{suffix}")
    with filepath.open("w") as f:
        f.write(content)
    return filepath


@app.route("/submit", methods=["POST"])
def submit() -> Response | str:
    assert contest
    data = request.get_json()

    task = contest.find_task_by_name(data["task"])
    if not task:
        return "Task not found"

    filepath = save_solution(data["solution"], data["lang"])

    ocimatic_path = Path(Path(__file__).parents[1], "bin", "ocimatic").resolve()
    cmd: list[Path | str] = [
        "python",
        ocimatic_path,
        "run",
        "--task",
        task.name,
        filepath,
    ]

    def stream() -> Iterator[str]:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        assert proc.stdout
        for line in proc.stdout:
            yield ansi2html(line)

    return Response(stream())


def run(contest_: core.Contest, port: int = 9999) -> None:
    global contest
    contest = contest_
    app.run(port=port, debug=True)
