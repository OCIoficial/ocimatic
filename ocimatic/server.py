from __future__ import annotations

import subprocess
import tempfile
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


contest: core.Contest | None = None
ocimatic_script_path = Path("ocimatic")
app = Flask(__name__)
app.secret_key = 'M\xf2\xba\xc0\xe3\xe55\xa0"\xff\x96\xba\xb8Jn\xc6#S\xa0t\xda\xb5[\r'


@app.route("/", methods=["POST", "GET"])
def server() -> str:
    assert contest
    return render_template("index.html", tasks=contest.tasks)


@app.route("/submit", methods=["POST"])
def submit() -> Response | str:
    assert contest
    data = request.get_json()

    task_name: str | None = data.get("task")
    if task_name is None:
        return "Select a task"

    task = contest.find_task_by_name(task_name)
    if not task:
        return "Task not found"

    ext = data["lang"]
    content = data.get("solution", "")

    def stream() -> Iterator[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir, f"solution.{ext}")
            with filepath.open("w") as f:
                f.write(content)

            cmd: list[Path | str] = [
                ocimatic_script_path,
                "run",
                "--task",
                task.name,
                filepath,
            ]

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
            assert proc.stdout
            for line in proc.stdout:
                yield ansi2html(line)

    return Response(stream())


def run(ocimatic_script_path_: Path, contest_: core.Contest, port: int = 9999) -> None:
    global contest
    global ocimatic_script_path
    contest = contest_
    ocimatic_script_path = ocimatic_script_path_
    app.run(port=port, debug=True)
