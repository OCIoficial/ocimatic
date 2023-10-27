import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from ansi2html import Ansi2HTMLConverter
from flask import Flask, Response, render_template, request
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

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


# def save_file(uploaded_file: FileStorage) -> Path:
#     dst_dir = upload_folder()
#     filename = secure_filename(uploaded_file.filename or "file")
#     filepath = Path(dst_dir, filename)
#     uploaded_file.save(str(Path(dst_dir, filename)))
#     return filepath


# def upload_solution() -> Path | None:
#     solution_text = request.form.get("solutionText")
#     if solution_text:
#         ext = request.form.get("lang")
#         dst_dir = upload_folder()
#         filepath = Path(dst_dir, f"solution.{ext}")
#         with filepath.open("w") as f:
#             f.write(solution_text)
#         return filepath

#     uploaded_file = cast(FileStorage, request.files.get("solutionFile"))
#     if not uploaded_file or uploaded_file.filename == "":
#         return None
#     return save_file(uploaded_file)


@app.route("/", methods=["POST", "GET"])
def server() -> str:
    assert contest
    return render_template("index.html", tasks=contest.tasks)


def save_solution(content: str, suffix: str) -> Path:
    dst_dir = upload_folder()
    filepath = Path(dst_dir, f"solution.{suffix}")
    with filepath.open("w") as f:
        f.write(content)
    return filepath


@app.route("/submit", methods=["POST"])
def submit() -> Response | str:
    assert contest
    data = request.get_json()
    filepath = save_solution(data["solution"], data["lang"])
    if not filepath:
        return "Unable to upload solution"

    task = contest.find_task(data["task"])
    if not task:
        return "Task not found"

    solution = task.load_solution_from_path(filepath)
    if not solution:
        return "Invalid solution"

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


def run(port: int = 9999) -> None:
    global contest
    contest_dir = core.find_contest_root()[0]
    contest = core.Contest(contest_dir)
    app.run(port=port, debug=True)
