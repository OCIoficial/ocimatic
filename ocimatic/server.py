import os
from importlib.util import find_spec
from io import StringIO

from ansi2html import Ansi2HTMLConverter
from flask import Flask, flash, redirect, render_template, request
from werkzeug.utils import secure_filename

import ocimatic
from ocimatic import core, filesystem, ui
from ocimatic.filesystem import Directory, FilePath


def ansi2html(ansi):
    return Ansi2HTMLConverter().convert(ansi)


UPLOAD_FOLDER = '/tmp/ocimatic/server/'

contest = None
app = Flask(__name__)
app.secret_key = 'M\xf2\xba\xc0\xe3\xe55\xa0"\xff\x96\xba\xb8Jn\xc6#S\xa0t\xda\xb5[\r'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def save_file(uploaded_file):
    dst_dir = filesystem.Directory(UPLOAD_FOLDER, create=True)
    filename = secure_filename(uploaded_file.filename)
    filepath = filesystem.FilePath(dst_dir, filename)
    uploaded_file.save(str(filesystem.FilePath(dst_dir, filename)))
    return filepath


def upload_solution(request):
    solution_text = request.form.get('solutionText')
    if solution_text:
        ext = request.form.get('lang')
        dst_dir = filesystem.Directory(UPLOAD_FOLDER, create=True)
        filepath = FilePath(dst_dir, f'solution.{ext}')
        with filepath.open('w') as f:
            f.write(solution_text)
        return filepath

    uploaded_file = request.files.get('solutionFile')
    if not uploaded_file or uploaded_file.filename == '':
        return False
    return save_file(uploaded_file)


@app.route('/', methods=['POST', 'GET'])
def server():
    result = ''
    if request.method == 'POST':
        filepath = upload_solution(request)
        if not filepath:
            flash('Please provide a solution')
            return redirect(request.url)
        task = contest.find_task(request.form.get('task'))
        solution = task.get_solution(filepath)
        if solution:
            stream = StringIO()
            with ui.capture_io(stream):
                task.run_solution(solution)
            result = stream.getvalue()
        else:
            result = 'Invalid solution'
    result = ansi2html(result)
    return render_template('index.html', tasks=contest.tasks, result=result)


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    return render_template('submit.html')


def run(port=9999):
    global contest  # pylint: disable=global-statement
    contest_dir = filesystem.change_directory()[0]
    contest = core.Contest(contest_dir)
    ocimatic.config['verbosity'] = 1
    app.run(port=port, debug=True)
